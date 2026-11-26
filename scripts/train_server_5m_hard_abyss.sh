#!/usr/bin/env bash
# Pure-PPO Two-Stage Curriculum Relay: Solo HARD (L9-14) -> Solo ABYSS (L15-20)
#
# Warm-starts directly from our strong Tier 2 (5-8) RL policy (96.9% on 5-8, 80.2% zero-shot on HARD,
# 21.7% zero-shot on ABYSS), completely bypassing bot.js BC (bot.js only achieves 24.2% on HARD and 3.7%
# on ABYSS). Anchors KL strictly to the previous-stage RL policy with fast annealing so early stability
# never caps late-stage breakthroughs.
#
# Stage 1 (hard):  mode_single_9..14  (HARD, 6 levels — 18 newly unlocked elites + R3/R9/R11 bounties/hardbuffs)
#   220 updates x 120 eps = 26,400 episodes (~2.6h on 60 workers + RTX 4090)
# Stage 2 (abyss): mode_single_15..20 (ABYSS, 6 levels — R1 1.11x/1.20x, R3 1.3x speed, R14 2.08x/3.58x, up to 4.37M Boss)
#   600 updates x 120 eps = 72,000 episodes (~7.2h on 60 workers + RTX 4090)
#
# Usage:
#   ./scripts/train_server_5m_hard_abyss.sh hard              # Stage 1 (HARD L9-14) only
#   ./scripts/train_server_5m_hard_abyss.sh abyss             # Stage 2 (ABYSS L15-20) only
#   ./scripts/train_server_5m_hard_abyss.sh all               # Full relay: hard -> abyss (~9.8h)
#   ./scripts/train_server_5m_hard_abyss.sh eval_hard [CKPT]  # 72-ep per-level eval on L9-14
#   ./scripts/train_server_5m_hard_abyss.sh eval_abyss [CKPT] # 72-ep per-level eval on L15-20

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
T2_BEST_CKPT="${T2_BEST_CKPT:-checkpoints/ppo_5m_solo_t2_best.pt}"

# Shared high-throughput AsyncActorPool & PPO optimizer settings
PPO_WORKERS="${PPO_WORKERS:-60}"
PPO_EPISODES_PER_UPDATE="${PPO_EPISODES_PER_UPDATE:-120}"
PPO_MIN_BATCH="${PPO_MIN_BATCH:-24}"
PPO_BATCH_TIMEOUT_MS="${PPO_BATCH_TIMEOUT_MS:-2.0}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-1024}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_CRITIC_LR="${PPO_CRITIC_LR:-2e-4}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.05}"

# Sub-step entropy weights (prevent 36-tile x 4-dir MOVE entropy hacking while preserving macro exploration)
PPO_SRC_ENT_WEIGHT="${PPO_SRC_ENT_WEIGHT:-0.5}"
PPO_TGT_ENT_WEIGHT="${PPO_TGT_ENT_WEIGHT:-0.15}"
PPO_OPT_ENT_WEIGHT="${PPO_OPT_ENT_WEIGHT:-0.15}"
PPO_STRATEGIC_ENT_MULT="${PPO_STRATEGIC_ENT_MULT:-4.0}"

# ---- Stage 1: HARD (L9-14, 6 levels) ---------------------------------------------------------------
HARD_UPDATES="${HARD_UPDATES:-220}"
HARD_CRITIC_WARMUP="${HARD_CRITIC_WARMUP:-3}"
HARD_LR="${HARD_LR:-2.5e-5}"
HARD_LR_END="${HARD_LR_END:-5e-6}"
HARD_ENT_COEF="${HARD_ENT_COEF:-0.004}"
HARD_ENT_COEF_END="${HARD_ENT_COEF_END:-0.001}"
HARD_KL_COEF="${HARD_KL_COEF:-0.03}"
HARD_KL_COEF_END="${HARD_KL_COEF_END:-0.002}"
HARD_KL_ANNEAL_UPDATES="${HARD_KL_ANNEAL_UPDATES:-80}"
HARD_MIX_RATIO_START="${HARD_MIX_RATIO_START:-0.15}"
HARD_MIX_RATIO_END="${HARD_MIX_RATIO_END:-0.0}"
HARD_MIX_ANNEAL_UPDATES="${HARD_MIX_ANNEAL_UPDATES:-30}"
HARD_ROUND_SURVIVE_BONUS="${HARD_ROUND_SURVIVE_BONUS:-0.04}"
HARD_HIDDEN_CLEAR_BONUS="${HARD_HIDDEN_CLEAR_BONUS:-0.6}"
HARD_SEED="${HARD_SEED:-400000}"
HARD_EVAL_EPISODES="${HARD_EVAL_EPISODES:-72}"  # 12 episodes x 6 HARD levels
HARD_EVAL_SEED_START="${HARD_EVAL_SEED_START:-29000}"
HARD_CKPT="${HARD_CKPT:-checkpoints/ppo_5m_solo_t3_hard_last.pt}"
HARD_BEST_CKPT="${HARD_BEST_CKPT:-checkpoints/ppo_5m_solo_t3_hard_best.pt}"

# ---- Stage 2: ABYSS (L15-20, 6 levels) -------------------------------------------------------------
ABYSS_UPDATES="${ABYSS_UPDATES:-600}"
ABYSS_CRITIC_WARMUP="${ABYSS_CRITIC_WARMUP:-4}"
ABYSS_LR="${ABYSS_LR:-3e-5}"
ABYSS_LR_END="${ABYSS_LR_END:-3e-6}"
ABYSS_ENT_COEF="${ABYSS_ENT_COEF:-0.0045}"
ABYSS_ENT_COEF_END="${ABYSS_ENT_COEF_END:-0.0005}"
ABYSS_KL_COEF="${ABYSS_KL_COEF:-0.03}"
ABYSS_KL_COEF_END="${ABYSS_KL_COEF_END:-0.001}"
ABYSS_KL_ANNEAL_UPDATES="${ABYSS_KL_ANNEAL_UPDATES:-120}"
ABYSS_MIX_RATIO_START="${ABYSS_MIX_RATIO_START:-0.30}"
ABYSS_MIX_RATIO_END="${ABYSS_MIX_RATIO_END:-0.0}"
ABYSS_MIX_ANNEAL_UPDATES="${ABYSS_MIX_ANNEAL_UPDATES:-100}"
ABYSS_ROUND_SURVIVE_BONUS="${ABYSS_ROUND_SURVIVE_BONUS:-0.05}"
ABYSS_HIDDEN_CLEAR_BONUS="${ABYSS_HIDDEN_CLEAR_BONUS:-0.8}"
ABYSS_SEED="${ABYSS_SEED:-600000}"
ABYSS_EVAL_EPISODES="${ABYSS_EVAL_EPISODES:-72}" # 12 episodes x 6 ABYSS levels
ABYSS_EVAL_SEED_START="${ABYSS_EVAL_SEED_START:-39000}"
ABYSS_CKPT="${ABYSS_CKPT:-checkpoints/ppo_5m_solo_t4_abyss_last.pt}"
ABYSS_BEST_CKPT="${ABYSS_BEST_CKPT:-checkpoints/ppo_5m_solo_t4_abyss_best.pt}"

EVAL_EVERY="${EVAL_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"

mkdir -p data checkpoints logs
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="${1:-all}"
LOG_FILE="${LOG_FILE:-logs/train_5m_hard_abyss_${STAGE}_${TS}.log}"

run_hard() {
  local init_ckpt="${HARD_INIT_CKPT:-${T2_BEST_CKPT}}"
  echo ">>> [1/2] Stage 1: Pure-PPO Curriculum on HARD (mode_single_9..14, ${HARD_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  echo "    init_ckpt=${init_ckpt} | kl=${HARD_KL_COEF}->${HARD_KL_COEF_END} (${HARD_KL_ANNEAL_UPDATES}u) | mix(tier2)=${HARD_MIX_RATIO_START}->${HARD_MIX_RATIO_END} (${HARD_MIX_ANNEAL_UPDATES}u)"
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier3 \
    --eval-modes tier3 \
    --mix-modes tier2 \
    --mix-ratio-start "${HARD_MIX_RATIO_START}" \
    --mix-ratio-end "${HARD_MIX_RATIO_END}" \
    --mix-anneal-updates "${HARD_MIX_ANNEAL_UPDATES}" \
    --init-ckpt "${init_ckpt}" \
    --updates "${HARD_UPDATES}" \
    --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" \
    --workers "${PPO_WORKERS}" \
    --min-batch "${PPO_MIN_BATCH}" \
    --batch-timeout-ms "${PPO_BATCH_TIMEOUT_MS}" \
    --minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --critic-warmup "${HARD_CRITIC_WARMUP}" \
    --lr "${HARD_LR}" \
    --lr-end "${HARD_LR_END}" \
    --critic-lr "${PPO_CRITIC_LR}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ent-coef "${HARD_ENT_COEF}" \
    --ent-coef-end "${HARD_ENT_COEF_END}" \
    --strategic-ent-mult "${PPO_STRATEGIC_ENT_MULT}" \
    --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" \
    --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}" \
    --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}" \
    --kl-coef "${HARD_KL_COEF}" \
    --kl-coef-end "${HARD_KL_COEF_END}" \
    --kl-anneal-updates "${HARD_KL_ANNEAL_UPDATES}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round 0.97 \
    --round-survive-bonus "${HARD_ROUND_SURVIVE_BONUS}" \
    --victory-reward 1.5 \
    --defeat-reward -1.5 \
    --hidden-clear-bonus "${HARD_HIDDEN_CLEAR_BONUS}" \
    --seed "${HARD_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${HARD_EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --eval-seed-start "${HARD_EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --out "${HARD_CKPT}" \
    --best-out "${HARD_BEST_CKPT}"
}

run_abyss() {
  local init_ckpt="${ABYSS_INIT_CKPT:-${HARD_BEST_CKPT}}"
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "WARN: ${init_ckpt} not found; falling back to ${T2_BEST_CKPT}" >&2
    init_ckpt="${T2_BEST_CKPT}"
  fi
  echo ">>> [2/2] Stage 2: Pure-PPO Curriculum on ABYSS (mode_single_15..20, ${ABYSS_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  echo "    init_ckpt=${init_ckpt} | kl=${ABYSS_KL_COEF}->${ABYSS_KL_COEF_END} (${ABYSS_KL_ANNEAL_UPDATES}u) | mix(tier3)=${ABYSS_MIX_RATIO_START}->${ABYSS_MIX_RATIO_END} (${ABYSS_MIX_ANNEAL_UPDATES}u)"
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier4 \
    --eval-modes tier4 \
    --mix-modes tier3 \
    --mix-ratio-start "${ABYSS_MIX_RATIO_START}" \
    --mix-ratio-end "${ABYSS_MIX_RATIO_END}" \
    --mix-anneal-updates "${ABYSS_MIX_ANNEAL_UPDATES}" \
    --init-ckpt "${init_ckpt}" \
    --updates "${ABYSS_UPDATES}" \
    --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" \
    --workers "${PPO_WORKERS}" \
    --min-batch "${PPO_MIN_BATCH}" \
    --batch-timeout-ms "${PPO_BATCH_TIMEOUT_MS}" \
    --minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --critic-warmup "${ABYSS_CRITIC_WARMUP}" \
    --lr "${ABYSS_LR}" \
    --lr-end "${ABYSS_LR_END}" \
    --critic-lr "${PPO_CRITIC_LR}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ent-coef "${ABYSS_ENT_COEF}" \
    --ent-coef-end "${ABYSS_ENT_COEF_END}" \
    --strategic-ent-mult "${PPO_STRATEGIC_ENT_MULT}" \
    --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" \
    --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}" \
    --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}" \
    --kl-coef "${ABYSS_KL_COEF}" \
    --kl-coef-end "${ABYSS_KL_COEF_END}" \
    --kl-anneal-updates "${ABYSS_KL_ANNEAL_UPDATES}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round 0.97 \
    --round-survive-bonus "${ABYSS_ROUND_SURVIVE_BONUS}" \
    --victory-reward 1.5 \
    --defeat-reward -1.5 \
    --hidden-clear-bonus "${ABYSS_HIDDEN_CLEAR_BONUS}" \
    --seed "${ABYSS_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${ABYSS_EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --eval-seed-start "${ABYSS_EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    --out "${ABYSS_CKPT}" \
    --best-out "${ABYSS_BEST_CKPT}"
}

run_eval() {
  local mode_preset="$1"
  local ckpt="$2"
  local eps="$3"
  local seed_start="$4"
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --eval-only "${ckpt}" \
    --modes "${mode_preset}" \
    --workers "${PPO_WORKERS}" \
    --eval-episodes "${eps}" \
    --eval-workers "${PPO_WORKERS}" \
    --eval-seed-start "${seed_start}" \
    --eval-temp "${EVAL_TEMP}" \
    --device "${DEVICE}"
}

{
  case "${STAGE}" in
    hard)       run_hard ;;
    abyss)      run_abyss ;;
    eval_hard)  run_eval tier3 "${2:-${HARD_BEST_CKPT}}" "${HARD_EVAL_EPISODES}" "${HARD_EVAL_SEED_START}" ;;
    eval_abyss) run_eval tier4 "${2:-${ABYSS_BEST_CKPT}}" "${ABYSS_EVAL_EPISODES}" "${ABYSS_EVAL_SEED_START}" ;;
    all)
      run_hard
      run_abyss
      ;;
    *)
      echo "Usage: $0 [hard|abyss|all|eval_hard [ckpt]|eval_abyss [ckpt]]" >&2
      exit 2
      ;;
  esac
  echo ">>> Done (${STAGE})."
} 2>&1 | tee -a "${LOG_FILE}"
