#!/usr/bin/env bash
# End-to-End 5.08M Solo Training Pipeline (Zero-Arch Feature Set v2, Anti-Collapse Curriculum)
#
# Trains from scratch on the enriched observation features (121 tokens x 12 cat x 28 cont, 6 sub-opts):
#   Stage 1 (bc):     Unified BC cold-start on mode_single_1..8 (ALL 10 maps, ALL 26 bonds / 10 core bonds,
#                     ALL 61 bands, R15 Hidden Core). Soft label-smoothing (0.10) + dropout (0.12) keep
#                     opening logits smooth and diverse before RL starts.
#   Stage 2 (normal): Masked PPO on mode_single_5..8 with L1..4 anchor replay (35% -> 8%).
#   Stage 3 (hard):   Masked PPO on mode_single_9..14 with L1..8 anchor replay (25% -> 8%).
#   Stage 4 (abyss):  Masked PPO on mode_single_15..20 with L1..14 anchor replay (30% -> 12%).
#
# Anti-collapse safeguards active across EVERY PPO stage:
#   L1-4 training episodes (including curriculum replay) mask weakness by default; eval stays unrestricted.
#   1. Unified 1-8 BC avoids 1-map / 18-bond T1-only early lock-in (egirShip & kazimierzShip active from step 0).
#   2. Decoupled --strategic-ent-coef protects opening BAND_DRAFT / INFO_CHECK / SP_DRAFT / CHOICE entropy
#      from 300:1 PREP-step dilution (watch `o_ent` in logs; healthy range ~0.6-2.5 during training).
#   3. Sub-step PREP entropy weights (src=0.5, tgt=0.15, opt=0.15) prevent 36x4 MOVE entropy hacking.
#   4. Intra-round GAE alignment (--round-discount --gamma-round 0.97 --gae-lambda-intra 0.99) gives Round-0
#      band/bond decisions real end-of-match credit instead of noisy early Critic estimates.
#   5. 5-update critic warmup + tighter --target-kl 0.04 + gentle LR (2.5e-5 -> 3e-6) prevent single-batch
#      opening spikes; KL anchor anneals cleanly to 0.0 so late-stage RL is never capped by earlier stages.
#   6. Execution hygiene (--prep-timeout-penalty -0.25 --repeat-state-penalty -0.015) from Stage 2 onward.
#
# Usage:
#   ./scripts/train_server_5m_v2.sh all                 # Full pipeline: bc -> normal -> hard -> abyss (~12.5h)
#   ./scripts/train_server_5m_v2.sh ppo                 # Stages 2-4: normal -> hard -> abyss (reuses BC ckpt)
#   ./scripts/train_server_5m_v2.sh bc                  # Stage 1: Unified 1-8 BC cold-start (~15m)
#   ./scripts/train_server_5m_v2.sh normal              # Stage 2: NORMAL (L5-8 + mix L1-4) PPO (~3.7h)
#   ./scripts/train_server_5m_v2.sh hard                # Stage 3: HARD (L9-14 + mix L1-8) PPO (~2.8h)
#   ./scripts/train_server_5m_v2.sh abyss               # Stage 4: ABYSS (L15-20 + mix L1-14) PPO (~6.0h)
#   ./scripts/train_server_5m_v2.sh eval_normal [CKPT]  # 64-ep eval on L5-8 (--no-break-cycles)
#   ./scripts/train_server_5m_v2.sh eval_hard   [CKPT]  # 72-ep eval on L9-14 (--no-break-cycles)
#   ./scripts/train_server_5m_v2.sh eval_abyss  [CKPT]  # 240-ep per-level & per-boss eval on L15-20
#
# All hyperparameters can be overridden via environment variables, e.g.:
#   DEVICE=cuda:1 ./scripts/train_server_5m_v2.sh all

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

# ---- Hardware & 5.08M Model Architecture (zero arch change) ----------------------------------------
DEVICE="${DEVICE:-cuda:0}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"
DIM_FF="${DIM_FF:-640}"
DROPOUT="${DROPOUT:-0.12}"

# ---- Shared High-Throughput AsyncActorPool & Anti-Collapse PPO Defaults ----------------------------
PPO_WORKERS="${PPO_WORKERS:-60}"
PPO_EPISODES_PER_UPDATE="${PPO_EPISODES_PER_UPDATE:-120}"
PPO_MIN_BATCH="${PPO_MIN_BATCH:-24}"
PPO_BATCH_TIMEOUT_MS="${PPO_BATCH_TIMEOUT_MS:-2.0}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-1024}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_CRITIC_LR="${PPO_CRITIC_LR:-2e-4}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.04}"
PPO_BAND_DIVERSITY_COEF="${PPO_BAND_DIVERSITY_COEF:-0.0}"
PPO_BAND_DIVERSITY_COEF_END="${PPO_BAND_DIVERSITY_COEF_END:--1}"
PPO_TIER1_BAN_BANDS="${PPO_TIER1_BAN_BANDS-band_weakness}"
PPO_GAMMA_ROUND="${PPO_GAMMA_ROUND:-0.97}"
PPO_GAE_LAMBDA_INTRA="${PPO_GAE_LAMBDA_INTRA:-0.99}"

PPO_STEP_PENALTY="${PPO_STEP_PENALTY:--0.0005}"
PPO_PREP_TIMEOUT_PENALTY="${PPO_PREP_TIMEOUT_PENALTY:--0.25}"
PPO_REPEAT_STATE_PENALTY="${PPO_REPEAT_STATE_PENALTY:--0.015}"

# Sub-step entropy shaping: prevent 36x4 MOVE entropy hacking while preserving macro exploration
PPO_SRC_ENT_WEIGHT="${PPO_SRC_ENT_WEIGHT:-0.5}"
PPO_TGT_ENT_WEIGHT="${PPO_TGT_ENT_WEIGHT:-0.15}"
PPO_OPT_ENT_WEIGHT="${PPO_OPT_ENT_WEIGHT:-0.15}"
PPO_STRATEGIC_ENT_MULT="${PPO_STRATEGIC_ENT_MULT:-4.0}"

# ---- Stage 1: Unified BC Cold-Start on mode_single_1..8 (150 eps x 8 levels = 1200 eps) ------------
BC_MODES="${BC_MODES:-1-8}"
BC_EVAL_MODES="${BC_EVAL_MODES:-tier2}"
BC_EPISODES="${BC_EPISODES:-1200}"
BC_WORKERS="${BC_WORKERS:-60}"
BC_ENCODE_WORKERS="${BC_ENCODE_WORKERS:-32}"
BC_EPOCHS="${BC_EPOCHS:-20}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-512}"
BC_LR="${BC_LR:-8e-4}"
BC_WEIGHT_DECAY="${BC_WEIGHT_DECAY:-1e-2}"
BC_LABEL_SMOOTHING="${BC_LABEL_SMOOTHING:-0.10}"  # softer opening logits prevent premature collapse in PPO
BC_VAL_RATIO="${BC_VAL_RATIO:-0.1}"
BC_SEED="${BC_SEED:-100000}"
BC_DATASET_CACHE="${BC_DATASET_CACHE:-data/v2feat_solo_1_8_${BC_EPISODES}ep.pt}"
BC_CKPT="${BC_CKPT:-checkpoints/v2feat_bc_5m_solo_1_8_${BC_EPISODES}ep.pt}"

# ---- Stage 2: NORMAL Curriculum PPO (mode_single_5..8 + L1..4 replay) ------------------------------
NORMAL_UPDATES="${NORMAL_UPDATES:-360}"
NORMAL_CRITIC_WARMUP="${NORMAL_CRITIC_WARMUP:-5}"
NORMAL_LR="${NORMAL_LR:-2.5e-5}"
NORMAL_LR_END="${NORMAL_LR_END:-4e-6}"
NORMAL_ENT_COEF="${NORMAL_ENT_COEF:-0.004}"
NORMAL_ENT_COEF_END="${NORMAL_ENT_COEF_END:-0.0008}"
# Note: strat_ent_bonus divides by n_strat (~1.4% of batch, ~70x amp vs policy_loss), so keep s_ent_c ~0.0004-0.0006
NORMAL_STRATEGIC_ENT_COEF="${NORMAL_STRATEGIC_ENT_COEF:-0.0006}"
NORMAL_STRATEGIC_ENT_COEF_END="${NORMAL_STRATEGIC_ENT_COEF_END:-0.00015}"
NORMAL_KL_COEF="${NORMAL_KL_COEF:-0.04}"
NORMAL_KL_COEF_END="${NORMAL_KL_COEF_END:-0.0}"
NORMAL_KL_ANNEAL_UPDATES="${NORMAL_KL_ANNEAL_UPDATES:-100}"
NORMAL_MIX_MODES="${NORMAL_MIX_MODES:-1-4}"
NORMAL_MIX_RATIO_START="${NORMAL_MIX_RATIO_START:-0.35}"
NORMAL_MIX_RATIO_END="${NORMAL_MIX_RATIO_END:-0.08}"
NORMAL_MIX_ANNEAL_UPDATES="${NORMAL_MIX_ANNEAL_UPDATES:-60}"
NORMAL_ROUND_SURVIVE_BONUS="${NORMAL_ROUND_SURVIVE_BONUS:-0.04}"
NORMAL_HIDDEN_CLEAR_BONUS="${NORMAL_HIDDEN_CLEAR_BONUS:-0.5}"
NORMAL_SEED="${NORMAL_SEED:-210000}"
NORMAL_EVAL_EPISODES="${NORMAL_EVAL_EPISODES:-64}"
NORMAL_EVAL_SEED_START="${NORMAL_EVAL_SEED_START:-19000}"
NORMAL_CKPT="${NORMAL_CKPT:-checkpoints/v2feat_ppo_5m_solo_t2_last.pt}"
NORMAL_BEST_CKPT="${NORMAL_BEST_CKPT:-checkpoints/v2feat_ppo_5m_solo_t2_best.pt}"

# ---- Stage 3: HARD Curriculum PPO (mode_single_9..14 + L1..8 replay) -------------------------------
HARD_UPDATES="${HARD_UPDATES:-240}"
HARD_CRITIC_WARMUP="${HARD_CRITIC_WARMUP:-5}"
HARD_LR="${HARD_LR:-2.5e-5}"
HARD_LR_END="${HARD_LR_END:-4e-6}"
HARD_ENT_COEF="${HARD_ENT_COEF:-0.004}"
HARD_ENT_COEF_END="${HARD_ENT_COEF_END:-0.0008}"
HARD_STRATEGIC_ENT_COEF="${HARD_STRATEGIC_ENT_COEF:-0.0005}"
HARD_STRATEGIC_ENT_COEF_END="${HARD_STRATEGIC_ENT_COEF_END:-0.00012}"
HARD_KL_COEF="${HARD_KL_COEF:-0.025}"
HARD_KL_COEF_END="${HARD_KL_COEF_END:-0.0}"
HARD_KL_ANNEAL_UPDATES="${HARD_KL_ANNEAL_UPDATES:-80}"
HARD_MIX_MODES="${HARD_MIX_MODES:-1-8}"
HARD_MIX_RATIO_START="${HARD_MIX_RATIO_START:-0.25}"
HARD_MIX_RATIO_END="${HARD_MIX_RATIO_END:-0.08}"
HARD_MIX_ANNEAL_UPDATES="${HARD_MIX_ANNEAL_UPDATES:-60}"
HARD_ROUND_SURVIVE_BONUS="${HARD_ROUND_SURVIVE_BONUS:-0.04}"
HARD_HIDDEN_CLEAR_BONUS="${HARD_HIDDEN_CLEAR_BONUS:-0.5}"
HARD_SEED="${HARD_SEED:-410000}"
HARD_EVAL_EPISODES="${HARD_EVAL_EPISODES:-72}"
HARD_EVAL_SEED_START="${HARD_EVAL_SEED_START:-29000}"
HARD_CKPT="${HARD_CKPT:-checkpoints/v2feat_ppo_5m_solo_t3_hard_last.pt}"
HARD_BEST_CKPT="${HARD_BEST_CKPT:-checkpoints/v2feat_ppo_5m_solo_t3_hard_best.pt}"

# ---- Stage 4: ABYSS Curriculum PPO (mode_single_15..20 + L1..14 replay) ----------------------------
ABYSS_UPDATES="${ABYSS_UPDATES:-500}"
ABYSS_CRITIC_WARMUP="${ABYSS_CRITIC_WARMUP:-5}"
ABYSS_LR="${ABYSS_LR:-2.5e-5}"
ABYSS_LR_END="${ABYSS_LR_END:-3e-6}"
ABYSS_ENT_COEF="${ABYSS_ENT_COEF:-0.004}"
ABYSS_ENT_COEF_END="${ABYSS_ENT_COEF_END:-0.0006}"
ABYSS_STRATEGIC_ENT_COEF="${ABYSS_STRATEGIC_ENT_COEF:-0.0004}"
ABYSS_STRATEGIC_ENT_COEF_END="${ABYSS_STRATEGIC_ENT_COEF_END:-0.00008}"
ABYSS_KL_COEF="${ABYSS_KL_COEF:-0.025}"
ABYSS_KL_COEF_END="${ABYSS_KL_COEF_END:-0.0}"
ABYSS_KL_ANNEAL_UPDATES="${ABYSS_KL_ANNEAL_UPDATES:-100}"
ABYSS_MIX_MODES="${ABYSS_MIX_MODES:-1-14}"
ABYSS_MIX_RATIO_START="${ABYSS_MIX_RATIO_START:-0.30}"
ABYSS_MIX_RATIO_END="${ABYSS_MIX_RATIO_END:-0.12}"
ABYSS_MIX_ANNEAL_UPDATES="${ABYSS_MIX_ANNEAL_UPDATES:-120}"
ABYSS_ROUND_SURVIVE_BONUS="${ABYSS_ROUND_SURVIVE_BONUS:-0.05}"
ABYSS_HIDDEN_CLEAR_BONUS="${ABYSS_HIDDEN_CLEAR_BONUS:-0.25}"
ABYSS_BOSS_DMG_DEFEAT_BONUS="${ABYSS_BOSS_DMG_DEFEAT_BONUS:-0.5}"
# Optional targeted overrides (left empty by default for pure general curriculum)
ABYSS_BOSS_WEIGHTS="${ABYSS_BOSS_WEIGHTS:-}"
ABYSS_BAN_BANDS="${ABYSS_BAN_BANDS:-}"
ABYSS_BAN_BAND_PROB="${ABYSS_BAN_BAND_PROB:-0.0}"
ABYSS_BAN_BOND_GROUPS="${ABYSS_BAN_BOND_GROUPS:-}"
ABYSS_BAN_BOND_GROUP_PROB="${ABYSS_BAN_BOND_GROUP_PROB:-0.0}"
ABYSS_SEED="${ABYSS_SEED:-610000}"
ABYSS_EVAL_EPISODES="${ABYSS_EVAL_EPISODES:-240}"
ABYSS_EVAL_SEED_START="${ABYSS_EVAL_SEED_START:-39000}"
ABYSS_CKPT="${ABYSS_CKPT:-checkpoints/v2feat_ppo_5m_solo_t4_abyss_last.pt}"
ABYSS_BEST_CKPT="${ABYSS_BEST_CKPT:-checkpoints/v2feat_ppo_5m_solo_t4_abyss_best.pt}"

# ---- Evaluation & Checkpointing --------------------------------------------------------------------
EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"
EVAL_BREAK_CYCLES="${EVAL_BREAK_CYCLES:-0}"  # 0 passes --no-break-cycles (exact deterministic scoring)
SAVE_EVERY="${SAVE_EVERY:-10}"

EVAL_EXTRA_ARGS=()
if [[ "${EVAL_BREAK_CYCLES}" == "0" ]]; then
  EVAL_EXTRA_ARGS+=(--no-break-cycles)
fi

mkdir -p data checkpoints logs
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="${1:-all}"
LOG_FILE="${LOG_FILE:-logs/train_5m_v2feat_${STAGE}_${TS}.log}"

preflight() {
  echo "========================================================================"
  echo "Stronghold Protocol 5.08M Solo Training — Zero-Arch Feature Set v2 (${STAGE})"
  echo "  ROOT_DIR       : ${ROOT_DIR}"
  echo "  PYTHON_BIN     : ${PYTHON_BIN}"
  echo "  STRONGHOLD_NODE: ${STRONGHOLD_NODE:-<PATH>}"
  echo "  DEVICE         : ${DEVICE}"
  echo "  MODEL          : d=${D_MODEL}, heads=${N_HEADS}, layers=${N_LAYERS}, ff=${DIM_FF}, bc_drop=${DROPOUT}"
  echo "  ANTI-COLLAPSE  : unified 1-8 BC (ls=${BC_LABEL_SMOOTHING}), decoupled s_ent_c, gae_lambda_intra=${PPO_GAE_LAMBDA_INTRA}, target_kl=${PPO_TARGET_KL}"
  echo "  LOG_FILE       : ${LOG_FILE}"
  echo "========================================================================"
  "${PYTHON_BIN}" -c "
import torch
from stronghold_sim.runtime import find_node
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import NUM_TOKENS, NUM_CAT_FIELDS, CONT_DIM, NUM_SUB_OPTS
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet

assert torch.cuda.is_available(), 'CUDA unavailable'
dev = torch.device('${DEVICE}')
idx = 0 if dev.index is None else dev.index
cat = get_catalog()
assert hasattr(cat, 'boss_pool_hp') and hasattr(cat, 'enemy_scale'), 'Missing v2 catalog tables!'
assert (NUM_TOKENS, NUM_CAT_FIELDS, CONT_DIM, NUM_SUB_OPTS) == (121, 12, 28, 6), f'Unexpected token shape: {(NUM_TOKENS, NUM_CAT_FIELDS, CONT_DIM, NUM_SUB_OPTS)}'
cfg = PolicyConfig(d_model=${D_MODEL}, n_heads=${N_HEADS}, n_layers=${N_LAYERS}, dim_ff=${DIM_FF}, dropout=${DROPOUT})
m = StrongholdPolicyNet(catalog=cat, config=cfg)
n = sum(p.numel() for p in m.parameters())
print('Node.js:', find_node())
print('PyTorch:', torch.__version__, '| GPU:', idx, torch.cuda.get_device_name(idx))
print(f'Catalog v2 OK: entities={cat.num_entities}, items={cat.num_items}, enemies={len(cat.enemy_meta)}, bosses={len(cat.boss_meta)}, core_bonds={len(cat.core_bond_indices)}')
print(f'Parameter count: {n:,} ({n/1e6:.2f}M) | obs=({NUM_TOKENS}, {NUM_CAT_FIELDS}, {CONT_DIM}) | sub_opts={NUM_SUB_OPTS}')
"
}

run_bc() {
  echo ">>> [Stage 1/4] Unified BC Cold-Start (modes=${BC_MODES}, ${BC_EPISODES} eps, ${BC_EPOCHS} epochs, ls=${BC_LABEL_SMOOTHING}, drop=${DROPOUT})..."
  "${PYTHON_BIN}" -u -m stronghold_rl.bc \
    --modes "${BC_MODES}" \
    --eval-modes "${BC_EVAL_MODES}" \
    --episodes "${BC_EPISODES}" \
    --workers "${BC_WORKERS}" \
    --encode-workers "${BC_ENCODE_WORKERS}" \
    --epochs "${BC_EPOCHS}" \
    --batch-size "${BC_BATCH_SIZE}" \
    --lr "${BC_LR}" \
    --weight-decay "${BC_WEIGHT_DECAY}" \
    --label-smoothing "${BC_LABEL_SMOOTHING}" \
    --val-ratio "${BC_VAL_RATIO}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --dim-ff "${DIM_FF}" \
    --dropout "${DROPOUT}" \
    --seed "${BC_SEED}" \
    --device "${DEVICE}" \
    --dataset-cache "${BC_DATASET_CACHE}" \
    --out "${BC_CKPT}" \
    --eval-episodes "${NORMAL_EVAL_EPISODES}" \
    --eval-workers "${BC_WORKERS}" \
    --eval-seed-start "${NORMAL_EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    "${EVAL_EXTRA_ARGS[@]}"
}

run_normal() {
  local init_ckpt="${NORMAL_INIT_CKPT:-${BC_CKPT}}"
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "ERROR: BC checkpoint not found at ${init_ckpt}. Run '$0 bc' first or set NORMAL_INIT_CKPT." >&2
    exit 1
  fi
  echo ">>> [Stage 2/4] NORMAL Curriculum PPO (tier2 L5-8 + mix ${NORMAL_MIX_MODES}, ${NORMAL_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  echo "    init=${init_ckpt} | s_ent_c=${NORMAL_STRATEGIC_ENT_COEF}->${NORMAL_STRATEGIC_ENT_COEF_END} | kl=${NORMAL_KL_COEF}->${NORMAL_KL_COEF_END} (${NORMAL_KL_ANNEAL_UPDATES}u)"
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier2 \
    --eval-modes tier2 \
    --mix-modes "${NORMAL_MIX_MODES}" \
    --mix-ratio-start "${NORMAL_MIX_RATIO_START}" \
    --mix-ratio-end "${NORMAL_MIX_RATIO_END}" \
    --mix-anneal-updates "${NORMAL_MIX_ANNEAL_UPDATES}" \
    --init-ckpt "${init_ckpt}" \
    --updates "${NORMAL_UPDATES}" \
    --episodes-per-update "${PPO_EPISODES_PER_UPDATE}" \
    --workers "${PPO_WORKERS}" \
    --min-batch "${PPO_MIN_BATCH}" \
    --batch-timeout-ms "${PPO_BATCH_TIMEOUT_MS}" \
    --minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --critic-warmup "${NORMAL_CRITIC_WARMUP}" \
    --lr "${NORMAL_LR}" \
    --lr-end "${NORMAL_LR_END}" \
    --critic-lr "${PPO_CRITIC_LR}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ent-coef "${NORMAL_ENT_COEF}" \
    --ent-coef-end "${NORMAL_ENT_COEF_END}" \
    --strategic-ent-mult "${PPO_STRATEGIC_ENT_MULT}" \
    --strategic-ent-coef "${NORMAL_STRATEGIC_ENT_COEF}" \
    --band-diversity-coef "${PPO_BAND_DIVERSITY_COEF}" \
    --band-diversity-coef-end "${PPO_BAND_DIVERSITY_COEF_END}" \
    --tier1-ban-bands "${PPO_TIER1_BAN_BANDS}" \
    --strategic-ent-coef-end "${NORMAL_STRATEGIC_ENT_COEF_END}" \
    --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" \
    --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}" \
    --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}" \
    --kl-coef "${NORMAL_KL_COEF}" \
    --kl-coef-end "${NORMAL_KL_COEF_END}" \
    --kl-anneal-updates "${NORMAL_KL_ANNEAL_UPDATES}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round "${PPO_GAMMA_ROUND}" \
    --gae-lambda-intra "${PPO_GAE_LAMBDA_INTRA}" \
    --step-penalty "${PPO_STEP_PENALTY}" \
    --prep-timeout-penalty "${PPO_PREP_TIMEOUT_PENALTY}" \
    --repeat-state-penalty "${PPO_REPEAT_STATE_PENALTY}" \
    --round-survive-bonus "${NORMAL_ROUND_SURVIVE_BONUS}" \
    --victory-reward 1.5 \
    --defeat-reward -1.5 \
    --hidden-clear-bonus "${NORMAL_HIDDEN_CLEAR_BONUS}" \
    --seed "${NORMAL_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${NORMAL_EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --eval-seed-start "${NORMAL_EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    "${EVAL_EXTRA_ARGS[@]}" \
    --out "${NORMAL_CKPT}" \
    --best-out "${NORMAL_BEST_CKPT}"
}

run_hard() {
  local init_ckpt="${HARD_INIT_CKPT:-${NORMAL_BEST_CKPT}}"
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "ERROR: NORMAL best checkpoint not found at ${init_ckpt}. Run '$0 normal' first or set HARD_INIT_CKPT." >&2
    exit 1
  fi
  echo ">>> [Stage 3/4] HARD Curriculum PPO (tier3 L9-14 + mix ${HARD_MIX_MODES}, ${HARD_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  echo "    init=${init_ckpt} | s_ent_c=${HARD_STRATEGIC_ENT_COEF}->${HARD_STRATEGIC_ENT_COEF_END} | kl=${HARD_KL_COEF}->${HARD_KL_COEF_END} (${HARD_KL_ANNEAL_UPDATES}u)"
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier3 \
    --eval-modes tier3 \
    --mix-modes "${HARD_MIX_MODES}" \
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
    --strategic-ent-coef "${HARD_STRATEGIC_ENT_COEF}" \
    --band-diversity-coef "${PPO_BAND_DIVERSITY_COEF}" \
    --band-diversity-coef-end "${PPO_BAND_DIVERSITY_COEF_END}" \
    --tier1-ban-bands "${PPO_TIER1_BAN_BANDS}" \
    --strategic-ent-coef-end "${HARD_STRATEGIC_ENT_COEF_END}" \
    --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" \
    --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}" \
    --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}" \
    --kl-coef "${HARD_KL_COEF}" \
    --kl-coef-end "${HARD_KL_COEF_END}" \
    --kl-anneal-updates "${HARD_KL_ANNEAL_UPDATES}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round "${PPO_GAMMA_ROUND}" \
    --gae-lambda-intra "${PPO_GAE_LAMBDA_INTRA}" \
    --step-penalty "${PPO_STEP_PENALTY}" \
    --prep-timeout-penalty "${PPO_PREP_TIMEOUT_PENALTY}" \
    --repeat-state-penalty "${PPO_REPEAT_STATE_PENALTY}" \
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
    "${EVAL_EXTRA_ARGS[@]}" \
    --out "${HARD_CKPT}" \
    --best-out "${HARD_BEST_CKPT}"
}

run_abyss() {
  local init_ckpt="${ABYSS_INIT_CKPT:-${HARD_BEST_CKPT}}"
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "ERROR: HARD best checkpoint not found at ${init_ckpt}. Run '$0 hard' first or set ABYSS_INIT_CKPT." >&2
    exit 1
  fi
  echo ">>> [Stage 4/4] ABYSS Curriculum PPO (tier4 L15-20 + mix ${ABYSS_MIX_MODES}, ${ABYSS_UPDATES} upd x ${PPO_EPISODES_PER_UPDATE} eps)..."
  echo "    init=${init_ckpt} | s_ent_c=${ABYSS_STRATEGIC_ENT_COEF}->${ABYSS_STRATEGIC_ENT_COEF_END} | kl=${ABYSS_KL_COEF}->${ABYSS_KL_COEF_END} (${ABYSS_KL_ANNEAL_UPDATES}u)"
  "${PYTHON_BIN}" -u -m stronghold_rl.ppo \
    --modes tier4 \
    --eval-modes tier4 \
    --mix-modes "${ABYSS_MIX_MODES}" \
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
    --strategic-ent-coef "${ABYSS_STRATEGIC_ENT_COEF}" \
    --band-diversity-coef "${PPO_BAND_DIVERSITY_COEF}" \
    --band-diversity-coef-end "${PPO_BAND_DIVERSITY_COEF_END}" \
    --tier1-ban-bands "${PPO_TIER1_BAN_BANDS}" \
    --strategic-ent-coef-end "${ABYSS_STRATEGIC_ENT_COEF_END}" \
    --ban-bands "${ABYSS_BAN_BANDS}" \
    --ban-band-prob "${ABYSS_BAN_BAND_PROB}" \
    --ban-bond-groups "${ABYSS_BAN_BOND_GROUPS}" \
    --ban-bond-group-prob "${ABYSS_BAN_BOND_GROUP_PROB}" \
    --src-ent-weight "${PPO_SRC_ENT_WEIGHT}" \
    --tgt-ent-weight "${PPO_TGT_ENT_WEIGHT}" \
    --opt-ent-weight "${PPO_OPT_ENT_WEIGHT}" \
    --kl-coef "${ABYSS_KL_COEF}" \
    --kl-coef-end "${ABYSS_KL_COEF_END}" \
    --kl-anneal-updates "${ABYSS_KL_ANNEAL_UPDATES}" \
    --target-kl "${PPO_TARGET_KL}" \
    --round-discount \
    --gamma-round "${PPO_GAMMA_ROUND}" \
    --gae-lambda-intra "${PPO_GAE_LAMBDA_INTRA}" \
    --step-penalty "${PPO_STEP_PENALTY}" \
    --prep-timeout-penalty "${PPO_PREP_TIMEOUT_PENALTY}" \
    --repeat-state-penalty "${PPO_REPEAT_STATE_PENALTY}" \
    --round-survive-bonus "${ABYSS_ROUND_SURVIVE_BONUS}" \
    --victory-reward 1.5 \
    --defeat-reward -1.5 \
    --hidden-clear-bonus "${ABYSS_HIDDEN_CLEAR_BONUS}" \
    --boss-dmg-defeat-bonus "${ABYSS_BOSS_DMG_DEFEAT_BONUS}" \
    --boss-weights "${ABYSS_BOSS_WEIGHTS}" \
    --seed "${ABYSS_SEED}" \
    --device "${DEVICE}" \
    --eval-episodes "${ABYSS_EVAL_EPISODES}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --eval-seed-start "${ABYSS_EVAL_SEED_START}" \
    --eval-temp "${EVAL_TEMP}" \
    "${EVAL_EXTRA_ARGS[@]}" \
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
    "${EVAL_EXTRA_ARGS[@]}" \
    --device "${DEVICE}"
}

{
  preflight
  case "${STAGE}" in
    bc)          run_bc ;;
    normal)      run_normal ;;
    hard)        run_hard ;;
    abyss)       run_abyss ;;
    eval_normal) run_eval tier2 "${2:-${NORMAL_BEST_CKPT}}" "${NORMAL_EVAL_EPISODES}" "${NORMAL_EVAL_SEED_START}" ;;
    eval_hard)   run_eval tier3 "${2:-${HARD_BEST_CKPT}}" "${HARD_EVAL_EPISODES}" "${HARD_EVAL_SEED_START}" ;;
    eval_abyss)  run_eval tier4 "${2:-${ABYSS_BEST_CKPT}}" "${ABYSS_EVAL_EPISODES}" "${ABYSS_EVAL_SEED_START}" ;;
    ppo)
      run_normal
      run_hard
      run_abyss
      ;;
    all)
      run_bc
      run_normal
      run_hard
      run_abyss
      ;;
    *)
      echo "Usage: $0 [all|ppo|bc|normal|hard|abyss|eval_normal [ckpt]|eval_hard [ckpt]|eval_abyss [ckpt]]" >&2
      exit 2
      ;;
  esac
  echo ">>> Done (${STAGE})."
} 2>&1 | tee -a "${LOG_FILE}"
