#!/usr/bin/env bash
# Fresh BC -> NORMAL -> HARD -> ABYSS, with random opening masks from rollout 1.
# Usage: bash scripts/train_server_diversity_scratch.sh [all|bc|ppo|normal|hard|abyss]
# Use the same RUN_ID when running stages separately; no old PPO weights are used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"
STAGE="${1:-all}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export LOG_FILE="${LOG_FILE:-logs/diversity_scratch_${RUN_ID}_${STAGE}.log}"
export BC_OPENING_POLICY="${BC_OPENING_POLICY:-balanced}"

# Start at full curriculum strength, including critic warmup. Half of episodes
# still have the full legal set so the policy learns unrestricted selection.
export PPO_GAE_LAMBDA_INTRA="${PPO_GAE_LAMBDA_INTRA:-1.0}"
export PPO_OPENING_MASK_PROB_START="${PPO_OPENING_MASK_PROB_START:-0.5}"
export PPO_OPENING_MASK_PROB_END="${PPO_OPENING_MASK_PROB_END:-${PPO_OPENING_MASK_PROB_START}}"
export PPO_OPENING_BAND_KEEP_START="${PPO_OPENING_BAND_KEEP_START:-0.35}"
export PPO_OPENING_BAND_KEEP_END="${PPO_OPENING_BAND_KEEP_END:-${PPO_OPENING_BAND_KEEP_START}}"
export PPO_OPENING_MASK_ANNEAL_UPDATES="${PPO_OPENING_MASK_ANNEAL_UPDATES:-1}"

export BC_CKPT="${BC_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_bc.pt}"
export NORMAL_CKPT="${NORMAL_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_normal_last.pt}"
export NORMAL_BEST_CKPT="${NORMAL_BEST_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_normal_best.pt}"
export HARD_CKPT="${HARD_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_hard_last.pt}"
export HARD_BEST_CKPT="${HARD_BEST_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_hard_best.pt}"
export ABYSS_CKPT="${ABYSS_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_abyss_last.pt}"
export ABYSS_BEST_CKPT="${ABYSS_BEST_CKPT:-checkpoints/diversity_scratch_${RUN_ID}_abyss_best.pt}"

# Deliberately replace inherited fine-tuning inputs with this run's stage chain.
# BC creates a new model and accepts only a cache matching its generation metadata.
export NORMAL_INIT_CKPT="${BC_CKPT}"
export HARD_INIT_CKPT="${NORMAL_BEST_CKPT}"
export ABYSS_INIT_CKPT="${HARD_BEST_CKPT}"
export NORMAL_SEED="${NORMAL_SEED:-2210000}"
export HARD_SEED="${HARD_SEED:-2410000}"
export ABYSS_SEED="${ABYSS_SEED:-2610000}"

OUTPUTS=()
case "${STAGE}" in
  all) OUTPUTS=("${BC_CKPT}" "${NORMAL_CKPT}" "${NORMAL_BEST_CKPT}" "${HARD_CKPT}" "${HARD_BEST_CKPT}" "${ABYSS_CKPT}" "${ABYSS_BEST_CKPT}") ;;
  bc) OUTPUTS=("${BC_CKPT}") ;;
  ppo) OUTPUTS=("${NORMAL_CKPT}" "${NORMAL_BEST_CKPT}" "${HARD_CKPT}" "${HARD_BEST_CKPT}" "${ABYSS_CKPT}" "${ABYSS_BEST_CKPT}") ;;
  normal) OUTPUTS=("${NORMAL_CKPT}" "${NORMAL_BEST_CKPT}") ;;
  hard) OUTPUTS=("${HARD_CKPT}" "${HARD_BEST_CKPT}") ;;
  abyss) OUTPUTS=("${ABYSS_CKPT}" "${ABYSS_BEST_CKPT}") ;;
  *)
    echo "Usage: $0 [all|bc|ppo|normal|hard|abyss]" >&2
    exit 2
    ;;
esac
for checkpoint in "${OUTPUTS[@]}"; do
  if [[ -e "${checkpoint}" ]]; then
    echo "ERROR: output already exists: ${checkpoint}. Use a new RUN_ID for a fresh run." >&2
    exit 1
  fi
done

exec bash "${SCRIPT_DIR}/train_server_5m_v2.sh" "${STAGE}"
