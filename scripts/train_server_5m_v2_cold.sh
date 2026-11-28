#!/usr/bin/env bash
# Pure PPO cold start for the enriched-feature model. No expert data or BC training.
# Random 5M weights -> L1-4 PPO -> existing L5-8 / L9-14 / L15-20 PPO curriculum.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: train_server_5m_v2_cold.sh [all|easy|normal|hard|abyss]

  all     Start from NEW random weights, then train easy -> normal -> hard -> abyss.
  easy    Start from NEW random weights and train only L1-4.
  normal  Train L5-8 from this run's easy_best checkpoint.
  hard    Train L9-14 from this run's normal_best checkpoint.
  abyss   Train L15-20 from this run's hard_best checkpoint.

Examples (from the repository root):
  RUN_ID=cold01 DEVICE=cuda:1 bash scripts/train_server_5m_v2_cold.sh all
  RUN_ID=cold01 bash scripts/train_server_5m_v2_cold.sh normal
  DRY_RUN=1 RUN_ID=preview bash scripts/train_server_5m_v2_cold.sh all

Use the same RUN_ID to continue at a later stage. Each invocation of a stage starts
a new optimizer/schedule; this is not an in-stage optimizer resume. all/easy refuse
to overwrite existing run checkpoints. Later stages use the previous stage's best
checkpoint, never the base launcher's BC_CKPT or *_INIT_CKPT environment defaults.

Defaults: EASY_UPDATES=600, NORMAL_UPDATES=360, HARD_UPDATES=240, ABYSS_UPDATES=500.
These are fixed budgets, not win-rate promotion thresholds. Tune after evaluation.
EASY_LR=1e-4 -> EASY_LR_END=2.5e-5; EASY_ENT_COEF=0.01 -> 0.003.
EASY_EVAL_EPISODES=64, INIT_SEED=110000, EASY_SEED=110000.
Shared PPO_*, evaluation, architecture, and later-stage overrides use the same
environment names as train_server_5m_v2.sh. Dropout is always zero for PPO.
DRY_RUN=1 prints the easy command and later-stage launcher configuration only;
it does not import Python, create files, or launch training. Full pipeline needs CUDA.
EOF
}

STAGE="${1:-all}"
case "${STAGE}" in
  -h|--help) usage; exit 0 ;;
  all|easy|normal|hard|abyss) ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  export PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
if [[ -z "${STRONGHOLD_NODE:-}" ]]; then
  for candidate in "${ROOT_DIR}/.tools/node/bin/node" "${ROOT_DIR}/.tools/node/node"; do
    if [[ -x "${candidate}" ]]; then export STRONGHOLD_NODE="${candidate}"; break; fi
  done
fi
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
PREFIX="checkpoints/v2feat_cold_${RUN_ID}"
INIT_CKPT="${INIT_CKPT:-${PREFIX}_init.pt}"
EASY_CKPT="${EASY_CKPT:-${PREFIX}_easy_last.pt}"
EASY_BEST_CKPT="${EASY_BEST_CKPT:-${PREFIX}_easy_best.pt}"
export NORMAL_CKPT="${NORMAL_CKPT:-${PREFIX}_normal_last.pt}"
export NORMAL_BEST_CKPT="${NORMAL_BEST_CKPT:-${PREFIX}_normal_best.pt}"
export HARD_CKPT="${HARD_CKPT:-${PREFIX}_hard_last.pt}"
export HARD_BEST_CKPT="${HARD_BEST_CKPT:-${PREFIX}_hard_best.pt}"
export ABYSS_CKPT="${ABYSS_CKPT:-${PREFIX}_abyss_last.pt}"
export ABYSS_BEST_CKPT="${ABYSS_BEST_CKPT:-${PREFIX}_abyss_best.pt}"
export NORMAL_INIT_CKPT="${EASY_BEST_CKPT}"
export HARD_INIT_CKPT="${NORMAL_BEST_CKPT}"
export ABYSS_INIT_CKPT="${HARD_BEST_CKPT}"
export LOG_FILE="${LOG_FILE:-logs/v2feat_cold_${RUN_ID}_${STAGE}.log}"

export DEVICE="${DEVICE:-cuda:0}"
export D_MODEL="${D_MODEL:-256}"
export N_HEADS="${N_HEADS:-8}"
export N_LAYERS="${N_LAYERS:-6}"
export DIM_FF="${DIM_FF:-640}"
export DROPOUT=0.0
export PPO_WORKERS="${PPO_WORKERS:-60}"
export PPO_EPISODES_PER_UPDATE="${PPO_EPISODES_PER_UPDATE:-120}"
export PPO_MIN_BATCH="${PPO_MIN_BATCH:-24}"
export PPO_BATCH_TIMEOUT_MS="${PPO_BATCH_TIMEOUT_MS:-2.0}"
export PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-1024}"
export PPO_EPOCHS="${PPO_EPOCHS:-2}"
export PPO_CRITIC_LR="${PPO_CRITIC_LR:-2e-4}"
export PPO_TARGET_KL="${PPO_TARGET_KL:-0.04}"
export PPO_GAMMA_ROUND="${PPO_GAMMA_ROUND:-0.97}"
export PPO_GAE_LAMBDA_INTRA="${PPO_GAE_LAMBDA_INTRA:-0.99}"
export PPO_STEP_PENALTY="${PPO_STEP_PENALTY:--0.0005}"
export PPO_PREP_TIMEOUT_PENALTY="${PPO_PREP_TIMEOUT_PENALTY:--0.25}"
export PPO_REPEAT_STATE_PENALTY="${PPO_REPEAT_STATE_PENALTY:--0.015}"
export PPO_SRC_ENT_WEIGHT="${PPO_SRC_ENT_WEIGHT:-0.5}"
export PPO_TGT_ENT_WEIGHT="${PPO_TGT_ENT_WEIGHT:-0.15}"
export PPO_OPT_ENT_WEIGHT="${PPO_OPT_ENT_WEIGHT:-0.15}"
export PPO_STRATEGIC_ENT_MULT="${PPO_STRATEGIC_ENT_MULT:-4.0}"
export EVAL_EVERY="${EVAL_EVERY:-20}"
export EVAL_TEMP="${EVAL_TEMP:-0.0}"
export EVAL_BREAK_CYCLES="${EVAL_BREAK_CYCLES:-0}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
INIT_SEED="${INIT_SEED:-110000}"

easy_command=(
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo
  --modes tier1 --eval-modes tier1 --init-ckpt "${INIT_CKPT}"
  --updates "${EASY_UPDATES:-600}"
  --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" --workers "${PPO_WORKERS}"
  --min-batch "${PPO_MIN_BATCH}" --batch-timeout-ms "${PPO_BATCH_TIMEOUT_MS}"
  --minibatch-size "${PPO_MINIBATCH_SIZE}" --ppo-epochs "${PPO_EPOCHS}"
  # A random reference is not a learned anchor; train the whole model immediately.
  --critic-warmup 0 --kl-coef 0.0 --kl-coef-end 0.0
  --lr "${EASY_LR:-1e-4}" --lr-end "${EASY_LR_END:-2.5e-5}"
  --critic-lr "${PPO_CRITIC_LR}" --target-kl "${PPO_TARGET_KL}"
  --ent-coef "${EASY_ENT_COEF:-0.01}" --ent-coef-end "${EASY_ENT_COEF_END:-0.003}"
  --strategic-ent-mult "${PPO_STRATEGIC_ENT_MULT}"
  --strategic-ent-coef "${EASY_STRATEGIC_ENT_COEF:-0.0006}"
  --strategic-ent-coef-end "${EASY_STRATEGIC_ENT_COEF_END:-0.00015}"
  --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}"
  --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}"
  --round-discount --gamma-round "${PPO_GAMMA_ROUND}" --gae-lambda-intra "${PPO_GAE_LAMBDA_INTRA}"
  --step-penalty "${PPO_STEP_PENALTY}" --prep-timeout-penalty "${PPO_PREP_TIMEOUT_PENALTY}"
  --repeat-state-penalty "${PPO_REPEAT_STATE_PENALTY}"
  --round-survive-bonus "${EASY_ROUND_SURVIVE_BONUS:-0.04}"
  --hidden-clear-bonus "${EASY_HIDDEN_CLEAR_BONUS:-0.5}"
  --victory-reward 1.5 --defeat-reward -1.5
  --seed "${EASY_SEED:-110000}" --device "${DEVICE}"
  --eval-episodes "${EASY_EVAL_EPISODES:-64}" --eval-every "${EVAL_EVERY}"
  --eval-seed-start "${EASY_EVAL_SEED_START:-9000}" --eval-temp "${EVAL_TEMP}"
  --save-every "${SAVE_EVERY}" --out "${EASY_CKPT}" --best-out "${EASY_BEST_CKPT}"
)
if [[ "${EVAL_BREAK_CYCLES}" == 0 ]]; then easy_command+=(--no-break-cycles); fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Pure PPO | RUN_ID=${RUN_ID} | stage=${STAGE} | LOG_FILE=${LOG_FILE}"
  if [[ "${STAGE}" == all || "${STAGE}" == easy ]]; then
    echo "Create NEW random checkpoint: ${INIT_CKPT} (seed=${INIT_SEED}, d=${D_MODEL}, heads=${N_HEADS}, layers=${N_LAYERS}, ff=${DIM_FF}, dropout=0)"
    printf '%q ' "${easy_command[@]}"; printf '\n'
  fi
  if [[ "${STAGE}" != easy ]]; then
    echo "Handoffs: normal=${NORMAL_INIT_CKPT}; hard=${HARD_INIT_CKPT}; abyss=${ABYSS_INIT_CKPT}"
    echo "Outputs: ${NORMAL_CKPT}; ${NORMAL_BEST_CKPT}; ${HARD_CKPT}; ${HARD_BEST_CKPT}; ${ABYSS_CKPT}; ${ABYSS_BEST_CKPT}"
    next_stage="${STAGE}"
    if [[ "${STAGE}" == all ]]; then next_stage=ppo; fi
    printf '%q ' bash "${SCRIPT_DIR}/train_server_5m_v2.sh" "${next_stage}"; printf '\n'
  fi
  exit 0
fi

if [[ "${STAGE}" == all || "${STAGE}" == easy ]]; then
  for path in "${INIT_CKPT}" "${EASY_CKPT}" "${EASY_BEST_CKPT}" \
    "${NORMAL_CKPT}" "${NORMAL_BEST_CKPT}" "${HARD_CKPT}" "${HARD_BEST_CKPT}" \
    "${ABYSS_CKPT}" "${ABYSS_BEST_CKPT}"; do
    if [[ -e "${path}" ]]; then
      echo "ERROR: ${path} exists. Use a new RUN_ID for a cold start, or select a later stage." >&2
      exit 1
    fi
  done
fi
mkdir -p "$(dirname "${LOG_FILE}")"
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null || echo 1024)" 2>/dev/null || true

run_easy() {
  echo "Pure PPO cold start | RUN_ID=${RUN_ID} | L1-4 | LOG_FILE=${LOG_FILE}"
  # Serialize fresh random weights to retain the intended model size: the PPO CLI
  # otherwise defaults to its smaller PolicyConfig. This does not run BC.
  "${PYTHON_BIN}" - "${INIT_CKPT}" "${INIT_SEED}" "${D_MODEL}" "${N_HEADS}" "${N_LAYERS}" "${DIM_FF}" "${DEVICE}" <<'PY'
import sys
from pathlib import Path
import torch
from stronghold_sim.runtime import find_node
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import NUM_TOKENS, NUM_CAT_FIELDS, CONT_DIM, NUM_SUB_OPTS
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet

path, seed, width, heads, layers, ff, device = sys.argv[1:]
assert (NUM_TOKENS, NUM_CAT_FIELDS, CONT_DIM, NUM_SUB_OPTS) == (121, 12, 28, 6), 'Expected enriched features'
if device.startswith('cuda'):
    torch.empty(1, device=device)  # Fail before writing anything if the selected GPU is unavailable.
print('Node.js:', find_node(), '| PyTorch:', torch.__version__, '| device:', device, flush=True)
torch.manual_seed(int(seed))
cfg = PolicyConfig(d_model=int(width), n_heads=int(heads), n_layers=int(layers), dim_ff=int(ff), dropout=0.0)
model = StrongholdPolicyNet(catalog=get_catalog(), config=cfg)
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('xb') as stream:
    torch.save({'config': cfg.to_dict(), 'state_dict': model.state_dict(),
                'extra': {'initialization': 'random', 'seed': int(seed)}}, stream)
print(f'Random initialization: {target} | {sum(p.numel() for p in model.parameters()):,} parameters | {cfg}', flush=True)
PY
  "${easy_command[@]}"
}

if [[ "${STAGE}" == all || "${STAGE}" == easy ]]; then
  run_easy 2>&1 | tee -a "${LOG_FILE}"
  if [[ "${STAGE}" == easy ]]; then exit 0; fi
  STAGE=ppo
fi
# The existing launcher's ppo/normal/hard/abyss routes never call run_bc.
# Later KL anchors and critic warmup refer only to this run's learned PPO weights.
exec bash "${SCRIPT_DIR}/train_server_5m_v2.sh" "${STAGE}"
