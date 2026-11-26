#!/usr/bin/env bash
# Smooth Curriculum Transfer Launcher: Solo Tier 1 (1-4) -> Solo Tier 2 (5-8, NORMAL)
#
# Designed to bridge the 4-way distribution shift from mode_single_1..4 to mode_single_5..8:
#   - 1 -> 10 battlefield maps (9 newly unlocked stage_emb entries)
#   - 18 -> 26 bonds (+egirShip, +kazimierzShip, +6 secondary bonds; +106 touched operators)
#   - Round 13-15 enemy scaling up to 1.18x ATK / 1.87x HP + bloodPointNormal Boss HP
#   - Round 15 Hidden Core (6 unseen hidden bosses)
#
# Usage:
#   ./scripts/train_server_5m_t2.sh zero_shot   # Zero-shot eval of Tier 1 checkpoint on 5-8
#   ./scripts/train_server_5m_t2.sh bridge      # Stage A: gentle mixed-tier BC warm-start for unseen entities/maps
#   ./scripts/train_server_5m_t2.sh ppo         # Stage B: exploration-enhanced Masked PPO with AsyncActorPool
#   ./scripts/train_server_5m_t2.sh all         # Full pipeline: zero_shot -> bridge -> ppo
#   ./scripts/train_server_5m_t2.sh eval [CKPT] # Held-out eval on mode_single_5..8 (64 unseen seeds)

set -euo pipefail

ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null || echo 1024)" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"

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

DEVICE="${DEVICE:-cuda:1}"
T1_CKPT="${T1_CKPT:-checkpoints/ppo_5m_solo_t1_best.pt}"
T1_DATA_CACHE="${T1_DATA_CACHE:-data/solo_t1_600ep.pt}"

# ---- Stage A: Light Mixed-Tier BC Bridge (warms up 9 maps + 8 bonds + 40+ operators + R15 bosses) ----
BRIDGE_EPISODES="${BRIDGE_EPISODES:-400}"
BRIDGE_WORKERS="${BRIDGE_WORKERS:-32}"
BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-8}"
BRIDGE_BATCH_SIZE="${BRIDGE_BATCH_SIZE:-512}"
BRIDGE_LR="${BRIDGE_LR:-2e-4}"
BRIDGE_LABEL_SMOOTHING="${BRIDGE_LABEL_SMOOTHING:-0.08}"  # softer target preserves exploration entropy for PPO
BRIDGE_DROPOUT="${BRIDGE_DROPOUT:-0.1}"
BRIDGE_MIX_RATIO="${BRIDGE_MIX_RATIO:-0.20}"              # 20% Tier 1 replay prevents catastrophic forgetting
BRIDGE_SEED="${BRIDGE_SEED:-11000}"
BRIDGE_CACHE="${BRIDGE_CACHE:-data/solo_t2_${BRIDGE_EPISODES}ep.pt}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/bc_5m_solo_t2_bridge.pt}"

# ---- Stage B: Exploration-Enhanced Masked PPO on mode_single_5..8 (~12h total budget) ---------------
# Enabled by AsyncActorPool (8x faster): 64 episodes/update (~20k steps, ~54s/update on 64-core + 4090)
# 750 updates x 64 eps = 48,000 episodes (~15M transitions) + 38 evals -> ~11.5 - 12.5 hours total
PPO_UPDATES="${PPO_UPDATES:-750}"
PPO_EPISODES_PER_UPDATE="${PPO_EPISODES_PER_UPDATE:-64}"
PPO_WORKERS="${PPO_WORKERS:-32}"
PPO_MIN_BATCH="${PPO_MIN_BATCH:-16}"
PPO_BATCH_TIMEOUT_MS="${PPO_BATCH_TIMEOUT_MS:-2.0}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-512}"
PPO_CRITIC_WARMUP="${PPO_CRITIC_WARMUP:-4}"
PPO_LR="${PPO_LR:-4e-5}"
PPO_LR_END="${PPO_LR_END:-5e-6}"
PPO_CRITIC_LR="${PPO_CRITIC_LR:-2e-4}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"

# Smooth curriculum mixing: 25% Tier 1 -> 0% over first 60 post-warmup updates (~1h bridge)
PPO_MIX_MODES="${PPO_MIX_MODES:-tier1}"
PPO_MIX_RATIO_START="${PPO_MIX_RATIO_START:-0.25}"
PPO_MIX_RATIO_END="${PPO_MIX_RATIO_END:-0.0}"
PPO_MIX_ANNEAL_UPDATES="${PPO_MIX_ANNEAL_UPDATES:-60}"

# Exploration & gradual un-tethering from bot.js reference policy across the 12h run
PPO_ENT_COEF="${PPO_ENT_COEF:-0.020}"
PPO_ENT_COEF_END="${PPO_ENT_COEF_END:-0.002}"
PPO_STRATEGIC_ENT_MULT="${PPO_STRATEGIC_ENT_MULT:-2.5}"
PPO_KL_COEF="${PPO_KL_COEF:-0.03}"
PPO_KL_COEF_END="${PPO_KL_COEF_END:-0.0}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.05}"

# Round-boundary discounting (gamma_intra=1.0, gamma_round=0.97) + terminal/Hidden Core reward shaping
PPO_GAMMA_ROUND="${PPO_GAMMA_ROUND:-0.97}"
PPO_ROUND_SURVIVE_BONUS="${PPO_ROUND_SURVIVE_BONUS:-0.04}"
PPO_VICTORY_REWARD="${PPO_VICTORY_REWARD:-1.5}"
PPO_DEFEAT_REWARD="${PPO_DEFEAT_REWARD:--1.5}"
PPO_HIDDEN_CLEAR_BONUS="${PPO_HIDDEN_CLEAR_BONUS:-0.5}"
PPO_SEED="${PPO_SEED:-30000}"

# ---- Evaluation & Artifacts -------------------------------------------------------------------------
EVAL_EPISODES="${EVAL_EPISODES:-64}"
EVAL_EVERY="${EVAL_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_SEED_START="${EVAL_SEED_START:-19000}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"

PPO_CKPT="${PPO_CKPT:-checkpoints/ppo_5m_solo_t2_last.pt}"
PPO_BEST_CKPT="${PPO_BEST_CKPT:-checkpoints/ppo_5m_solo_t2_best.pt}"

mkdir -p data checkpoints logs
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="${1:-all}"
LOG_FILE="${LOG_FILE:-logs/train_5m_t2_${STAGE}_${TS}.log}"

run_zero_shot() {
  echo ">>> [0/2] Zero-shot eval of Tier 1 checkpoint (${T1_CKPT}) on mode_single_5..8 (${EVAL_EPISODES} seeds)..."
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --eval-only "${T1_CKPT}" \
    --modes tier2 \
    --workers "${BRIDGE_WORKERS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-workers "${BRIDGE_WORKERS}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --device "${DEVICE}"
}

run_bridge() {
  echo ">>> [1/2] Stage A: Mixed-Tier BC Bridge (${BRIDGE_EPISODES} t2 eps + ${BRIDGE_MIX_RATIO} t1 replay, ${BRIDGE_EPOCHS} ep, ls=${BRIDGE_LABEL_SMOOTHING})..."
  local mix_args=()
  if [[ -f "${T1_DATA_CACHE}" ]]; then
    mix_args=(--mix-dataset-cache "${T1_DATA_CACHE}" --mix-ratio "${BRIDGE_MIX_RATIO}")
  fi
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --modes tier2 \
    --init-ckpt "${T1_CKPT}" \
    --episodes "${BRIDGE_EPISODES}" \
    --workers "${BRIDGE_WORKERS}" \
    --encode-workers "${BRIDGE_WORKERS}" \
    --epochs "${BRIDGE_EPOCHS}" \
    --batch-size "${BRIDGE_BATCH_SIZE}" \
    --lr "${BRIDGE_LR}" \
    --weight-decay 1e-2 \
    --label-smoothing "${BRIDGE_LABEL_SMOOTHING}" \
    --dropout "${BRIDGE_DROPOUT}" \
    --val-ratio 0.1 \
    --seed "${BRIDGE_SEED}" \
    --device "${DEVICE}" \
    --dataset-cache "${BRIDGE_CACHE}" \
    "${mix_args[@]}" \
    --out "${BRIDGE_CKPT}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-workers "${BRIDGE_WORKERS}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}"
}

run_ppo() {
  local init_ckpt="${INIT_CKPT:-${BRIDGE_CKPT}}"
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "WARN: ${init_ckpt} not found; falling back to direct transfer from ${T1_CKPT}" >&2
    init_ckpt="${T1_CKPT}"
  fi
  echo ">>> [2/2] Stage B: Exploration-Enhanced Masked PPO on mode_single_5..8 (${PPO_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier2 \
    --eval-modes tier2 \
    --mix-modes "${PPO_MIX_MODES}" \
    --mix-ratio-start "${PPO_MIX_RATIO_START}" \
    --mix-ratio-end "${PPO_MIX_RATIO_END}" \
    --mix-anneal-updates "${PPO_MIX_ANNEAL_UPDATES}" \
    --init-ckpt "${init_ckpt}" \
    --updates "${PPO_UPDATES}" \
    --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" \
    --workers "${PPO_WORKERS}" \
    --min-batch "${PPO_MIN_BATCH}" \
    --batch-timeout-ms "${PPO_BATCH_TIMEOUT_MS}" \
    --minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --critic-warmup "${PPO_CRITIC_WARMUP}" \
    --lr "${PPO_LR}" \
    --lr-end "${PPO_LR_END}" \
    --critic-lr "${PPO_CRITIC_LR}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ent-coef "${PPO_ENT_COEF}" \
    --ent-coef-end "${PPO_ENT_COEF_END}" \
    --strategic-ent-mult "${PPO_STRATEGIC_ENT_MULT}" \
    --kl-coef "${PPO_KL_COEF}" \
    --kl-coef-end "${PPO_KL_COEF_END}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round "${PPO_GAMMA_ROUND}" \
    --round-survive-bonus "${PPO_ROUND_SURVIVE_BONUS}" \
    --victory-reward "${PPO_VICTORY_REWARD}" \
    --defeat-reward "${PPO_DEFEAT_REWARD}" \
    --hidden-clear-bonus "${PPO_HIDDEN_CLEAR_BONUS}" \
    --seed "${PPO_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --out "${PPO_CKPT}" \
    --best-out "${PPO_BEST_CKPT}"
}

run_eval() {
  local ckpt="${2:-${PPO_BEST_CKPT}}"
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --eval-only "${ckpt}" \
    --modes tier2 \
    --workers "${BRIDGE_WORKERS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-workers "${BRIDGE_WORKERS}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --device "${DEVICE}"
}

{
  case "${STAGE}" in
    zero_shot) run_zero_shot ;;
    bridge)    run_bridge ;;
    ppo)       run_ppo ;;
    eval)      run_eval "$@" ;;
    all)
      run_zero_shot
      run_bridge
      run_ppo
      ;;
    *)
      echo "Usage: $0 [zero_shot|bridge|ppo|eval [ckpt]|all]" >&2
      exit 2
      ;;
  esac
  echo ">>> Done (${STAGE})."
} 2>&1 | tee -a "${LOG_FILE}"
