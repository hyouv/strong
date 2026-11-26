#!/usr/bin/env bash
# Large-scale 5M-parameter solo training launcher for Stronghold Protocol (act3autochess@V3_1)
# Target hardware: multi-core CPU + NVIDIA GPU (e.g., 64-thread Xeon + RTX 4090)
#
# Usage:
#   ./scripts/train_server_5m.sh           # Run full pipeline: Stage 1 (BC) -> Stage 2 (Masked PPO)
#   ./scripts/train_server_5m.sh bc        # Run Stage 1 (Expert collection + 5M BC cold-start) only
#   ./scripts/train_server_5m.sh ppo       # Run Stage 2 (Masked PPO from BC checkpoint) only
#   ./scripts/train_server_5m.sh eval [CKPT_PATH]  # Evaluate a checkpoint on 32 held-out solo seeds
#
# All hyperparameters can be overridden via environment variables, e.g.:
#   BC_EPISODES=800 PPO_UPDATES=100 ./scripts/train_server_5m.sh

set -euo pipefail

# Raise soft open-file limit if permitted (defense-in-depth for 32+ workers)
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null || echo 1024)" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"

# ---- Locate project-local Python & Node.js runtimes ------------------------
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ -z "${STRONGHOLD_NODE:-}" ]]; then
  for cand in \
    "${ROOT_DIR}/.tools/node/bin/node" \
    "${ROOT_DIR}/.tools/node/node"; do
    if [[ -x "${cand}" ]]; then
      export STRONGHOLD_NODE="${cand}"
      break
    fi
  done
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# ---- Hardware & 5M Model Architecture (5,079,513 params = 5.08M) -----------
DEVICE="${DEVICE:-cuda:1}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"
DIM_FF="${DIM_FF:-640}"
DROPOUT="${DROPOUT:-0.0}"

# ---- Stage 1: Solo Tier 1-4 (8 core bonds) Expert Collection + BC ----------
BC_EPISODES="${BC_EPISODES:-600}"
BC_WORKERS="${BC_WORKERS:-32}"
BC_ENCODE_WORKERS="${BC_ENCODE_WORKERS:-32}"
BC_EPOCHS="${BC_EPOCHS:-30}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-512}"
BC_LR="${BC_LR:-8e-4}"
BC_VAL_RATIO="${BC_VAL_RATIO:-0.1}"
BC_SEED="${BC_SEED:-1000}"

# ---- Stage 2: Masked PPO Fine-Tuning (anchored to BC) ----------------------
PPO_UPDATES="${PPO_UPDATES:-60}"
PPO_EPISODES_PER_UPDATE="${PPO_EPISODES_PER_UPDATE:-32}"
PPO_WORKERS="${PPO_WORKERS:-32}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-512}"
PPO_CRITIC_WARMUP="${PPO_CRITIC_WARMUP:-3}"
PPO_LR="${PPO_LR:-3e-5}"
PPO_CRITIC_LR="${PPO_CRITIC_LR:-2e-4}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_KL_COEF="${PPO_KL_COEF:-0.05}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.05}"
PPO_SEED="${PPO_SEED:-20000}"

# ---- Evaluation & Artifacts ------------------------------------------------
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_SEED_START="${EVAL_SEED_START:-9000}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"

DATASET_CACHE="${DATASET_CACHE:-data/solo_t1_${BC_EPISODES}ep.pt}"
BC_CKPT="${BC_CKPT:-checkpoints/bc_5m_solo_t1_${BC_EPISODES}ep.pt}"
PPO_CKPT="${PPO_CKPT:-checkpoints/ppo_5m_solo_t1_last.pt}"
PPO_BEST_CKPT="${PPO_BEST_CKPT:-checkpoints/ppo_5m_solo_t1_best.pt}"

mkdir -p data checkpoints logs
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="${1:-all}"
LOG_FILE="${LOG_FILE:-logs/train_5m_${STAGE}_${TS}.log}"

preflight() {
  echo "============================================================"
  echo "Stronghold Protocol 5M Solo Training (${STAGE})"
  echo "  ROOT_DIR      : ${ROOT_DIR}"
  echo "  PYTHON_BIN    : ${PYTHON_BIN}"
  echo "  STRONGHOLD_NODE: ${STRONGHOLD_NODE:-<PATH>}"
  echo "  DEVICE        : ${DEVICE}"
  echo "  MODEL         : d=${D_MODEL}, heads=${N_HEADS}, layers=${N_LAYERS}, ff=${DIM_FF}, dropout=${DROPOUT}"
  echo "  LOG_FILE      : ${LOG_FILE}"
  echo "============================================================"
  "${PYTHON_BIN}" -c "
import torch
from stronghold_sim.runtime import find_node
from stronghold_rl.catalog import get_catalog
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet
assert torch.cuda.is_available(), 'CUDA unavailable'
dev = torch.device('${DEVICE}')
idx = 0 if dev.index is None else dev.index
print('Node.js:', find_node())
print('PyTorch:', torch.__version__, '| GPU:', idx, torch.cuda.get_device_name(idx))
cfg = PolicyConfig(d_model=${D_MODEL}, n_heads=${N_HEADS}, n_layers=${N_LAYERS}, dim_ff=${DIM_FF}, dropout=${DROPOUT})
m = StrongholdPolicyNet(catalog=get_catalog(), config=cfg)
n = sum(p.numel() for p in m.parameters())
print(f'Parameter count: {n:,} ({n/1e6:.2f}M)')
"
}

run_bc() {
  echo ">>> [Stage 1/2] Starting 5M BC cold-start (${BC_EPISODES} solo eps, ${BC_EPOCHS} epochs on ${DEVICE})..."
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --episodes "${BC_EPISODES}" \
    --workers "${BC_WORKERS}" \
    --encode-workers "${BC_ENCODE_WORKERS}" \
    --epochs "${BC_EPOCHS}" \
    --batch-size "${BC_BATCH_SIZE}" \
    --lr "${BC_LR}" \
    --val-ratio "${BC_VAL_RATIO}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --dim-ff "${DIM_FF}" \
    --dropout "${DROPOUT}" \
    --seed "${BC_SEED}" \
    --device "${DEVICE}" \
    --dataset-cache "${DATASET_CACHE}" \
    --out "${BC_CKPT}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-workers "${BC_WORKERS}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}"
}

run_ppo() {
  if [[ ! -f "${BC_CKPT}" ]]; then
    echo "ERROR: BC checkpoint not found at ${BC_CKPT}. Run './scripts/train_server_5m.sh bc' first or set BC_CKPT." >&2
    exit 1
  fi
  echo ">>> [Stage 2/2] Starting 5M Masked PPO (${PPO_UPDATES} updates x ${PPO_EPISODES_PER_UPDATE} eps on ${DEVICE})..."
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --init-ckpt "${BC_CKPT}" \
    --updates "${PPO_UPDATES}" \
    --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" \
    --workers "${PPO_WORKERS}" \
    --minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --critic-warmup "${PPO_CRITIC_WARMUP}" \
    --lr "${PPO_LR}" \
    --critic-lr "${PPO_CRITIC_LR}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --kl-coef "${PPO_KL_COEF}" \
    --target-kl "${PPO_TARGET_KL}" \
    --seed "${PPO_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --out "${PPO_CKPT}" \
    --best-out "${PPO_BEST_CKPT}"
}

run_eval() {
  local ckpt="${2:-${PPO_BEST_CKPT}}"
  echo ">>> Evaluating ${ckpt} on ${EVAL_EPISODES} held-out solo seeds (${DEVICE})..."
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --eval-only "${ckpt}" \
    --workers "${BC_WORKERS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-workers "${BC_WORKERS}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --device "${DEVICE}"
}

{
  preflight
  case "${STAGE}" in
    all)
      run_bc
      run_ppo
      ;;
    bc)
      run_bc
      ;;
    ppo)
      run_ppo
      ;;
    eval)
      run_eval "$@"
      ;;
    *)
      echo "Usage: $0 [all|bc|ppo|eval [ckpt]]" >&2
      exit 2
      ;;
  esac
  echo ">>> Done (${STAGE})."
} 2>&1 | tee -a "${LOG_FILE}"
