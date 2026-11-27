#!/usr/bin/env bash
# Fine-tune an existing v2 checkpoint with a gradual random-opening curriculum.
# Usage: bash scripts/train_server_diversity.sh [normal|hard|abyss]
# Override INIT_CKPT / RUN_ID / DEVICE and any train_server_5m_v2.sh environment option.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"
STAGE="${1:-normal}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export LOG_FILE="${LOG_FILE:-logs/diversity_${STAGE}_${RUN_ID}.log}"

# Warmup + first actor update are unrestricted. Reach the final restriction on
# actor update 120, then hold it; half of episodes remain fully unrestricted.
export PPO_GAE_LAMBDA_INTRA="${PPO_GAE_LAMBDA_INTRA:-1.0}"
export PPO_OPENING_MASK_PROB_START="${PPO_OPENING_MASK_PROB_START:-0.0}"
export PPO_OPENING_MASK_PROB_END="${PPO_OPENING_MASK_PROB_END:-0.5}"
export PPO_OPENING_BAND_KEEP_START="${PPO_OPENING_BAND_KEEP_START:-1.0}"
export PPO_OPENING_BAND_KEEP_END="${PPO_OPENING_BAND_KEEP_END:-0.35}"
export PPO_OPENING_MASK_ANNEAL_UPDATES="${PPO_OPENING_MASK_ANNEAL_UPDATES:-120}"

# These are weight-only fine-tuning runs: a fresh optimizer and a short critic
# warmup, with new seeds and separate output paths preserving the input model.
case "${STAGE}" in
  normal)
    export NORMAL_INIT_CKPT="${INIT_CKPT:-${NORMAL_INIT_CKPT:-checkpoints/v2feat_ppo_5m_solo_t2_best.pt}}"
    export NORMAL_CKPT="${NORMAL_CKPT:-checkpoints/diversity_normal_${RUN_ID}_last.pt}"
    export NORMAL_BEST_CKPT="${NORMAL_BEST_CKPT:-checkpoints/diversity_normal_${RUN_ID}_best.pt}"
    export NORMAL_UPDATES="${NORMAL_UPDATES:-240}"
    export NORMAL_LR="${NORMAL_LR:-1e-5}"
    export NORMAL_LR_END="${NORMAL_LR_END:-3e-6}"
    export NORMAL_KL_COEF="${NORMAL_KL_COEF:-0.01}"
    export NORMAL_KL_ANNEAL_UPDATES="${NORMAL_KL_ANNEAL_UPDATES:-60}"
    export NORMAL_MIX_RATIO_START="${NORMAL_MIX_RATIO_START:-0.08}"
    export NORMAL_SEED="${NORMAL_SEED:-1210000}"
    ;;
  hard)
    export HARD_INIT_CKPT="${INIT_CKPT:-${HARD_INIT_CKPT:-checkpoints/v2feat_ppo_5m_solo_t3_hard_best.pt}}"
    export HARD_CKPT="${HARD_CKPT:-checkpoints/diversity_hard_${RUN_ID}_last.pt}"
    export HARD_BEST_CKPT="${HARD_BEST_CKPT:-checkpoints/diversity_hard_${RUN_ID}_best.pt}"
    export HARD_LR="${HARD_LR:-1e-5}"
    export HARD_LR_END="${HARD_LR_END:-3e-6}"
    export HARD_KL_COEF="${HARD_KL_COEF:-0.01}"
    export HARD_KL_ANNEAL_UPDATES="${HARD_KL_ANNEAL_UPDATES:-60}"
    export HARD_MIX_RATIO_START="${HARD_MIX_RATIO_START:-0.08}"
    export HARD_SEED="${HARD_SEED:-1410000}"
    ;;
  abyss)
    export ABYSS_INIT_CKPT="${INIT_CKPT:-${ABYSS_INIT_CKPT:-checkpoints/v2feat_ppo_5m_solo_t4_abyss_best.pt}}"
    export ABYSS_CKPT="${ABYSS_CKPT:-checkpoints/diversity_abyss_${RUN_ID}_last.pt}"
    export ABYSS_BEST_CKPT="${ABYSS_BEST_CKPT:-checkpoints/diversity_abyss_${RUN_ID}_best.pt}"
    export ABYSS_LR="${ABYSS_LR:-1e-5}"
    export ABYSS_LR_END="${ABYSS_LR_END:-3e-6}"
    export ABYSS_KL_COEF="${ABYSS_KL_COEF:-0.01}"
    export ABYSS_KL_ANNEAL_UPDATES="${ABYSS_KL_ANNEAL_UPDATES:-60}"
    export ABYSS_MIX_RATIO_START="${ABYSS_MIX_RATIO_START:-0.12}"
    export ABYSS_SEED="${ABYSS_SEED:-1610000}"
    ;;
  *)
    echo "Usage: $0 [normal|hard|abyss]" >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/train_server_5m_v2.sh" "${STAGE}"
