"""Transformer encoder (121 semantic tokens, isolated embeddings) + 4-step Autoregressive Pointer Head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .catalog import GameCatalog, get_catalog
from .codec import (
    CAT_BAND,
    CAT_BOND,
    CAT_DIR,
    CAT_ENTITY,
    CAT_EQUIP0,
    CAT_SEG,
    CAT_STAGE,
    CONT_DIM,
    MAX_EQUIP_SLOTS,
    NO_ARG,
    NUM_ACTION_TYPES,
    NUM_SEGMENTS,
    NUM_SUB_OPTS,
    NUM_TOKENS,
    POLICY_CLS_IDX,
    TYPE_BAND,
    TYPE_EQUIP,
    VALUE_CLS_IDX,
    ActionMaskSpec,
    ActionTree,
    EncodedFrame,
)


@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class StepMasks:
    """Flattened per-step legal masks along a single realized composite action path."""

    type_mask: torch.Tensor   # [NUM_ACTION_TYPES] bool
    src_mask: torch.Tensor    # [NUM_TOKENS] bool
    has_src: bool
    band_mask: torch.Tensor   # [num_bands] bool
    has_band: bool
    tgt_mask: torch.Tensor    # [NUM_TOKENS] bool
    has_tgt: bool
    opt_mask: torch.Tensor    # [NUM_SUB_OPTS] bool
    has_opt: bool


def extract_step_masks(enc: EncodedFrame, a_type: int, a_src: int, a_tgt: int, a_opt: int) -> StepMasks:
    """Extract exact hierarchical masks along the realized path `(a_type, a_src, a_tgt, a_opt)`."""
    tree = enc.action_tree
    type_mask = tree.type_mask

    has_band = a_type == TYPE_BAND and a_src != NO_ARG
    has_src = a_type != TYPE_BAND and a_src != NO_ARG
    has_tgt = a_tgt != NO_ARG
    has_opt = a_opt != NO_ARG

    if has_band:
        bm = tree.src_mask(TYPE_BAND)
        band_mask = bm if bm is not None else torch.zeros(tree.num_bands, dtype=torch.bool)
        src_mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    elif has_src:
        sm = tree.src_mask(a_type)
        src_mask = sm if sm is not None else torch.zeros(NUM_TOKENS, dtype=torch.bool)
        band_mask = torch.zeros(tree.num_bands, dtype=torch.bool)
    else:
        src_mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
        band_mask = torch.zeros(tree.num_bands, dtype=torch.bool)

    if has_tgt:
        tm = tree.tgt_mask(a_type, a_src)
        tgt_mask = tm if tm is not None else torch.zeros(NUM_TOKENS, dtype=torch.bool)
    else:
        tgt_mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)

    if has_opt:
        om = tree.opt_mask(a_type, a_src, a_tgt)
        opt_mask = om if om is not None else torch.zeros(NUM_SUB_OPTS, dtype=torch.bool)
    else:
        opt_mask = torch.zeros(NUM_SUB_OPTS, dtype=torch.bool)

    return StepMasks(
        type_mask=type_mask,
        src_mask=src_mask,
        has_src=has_src,
        band_mask=band_mask,
        has_band=has_band,
        tgt_mask=tgt_mask,
        has_tgt=has_tgt,
        opt_mask=opt_mask,
        has_opt=has_opt,
    )


def _masked_log_softmax_and_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically safe masked log_softmax and entropy (zero NaNs even on all-False inactive rows)."""
    safe_mask = mask.clone()
    if active is not None:
        inactive = ~active
        if inactive.any():
            safe_mask[inactive, 0] = True
    masked = logits.masked_fill(~safe_mask, -1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    probs = log_probs.exp()
    ent = -(probs * log_probs.clamp(min=-1e4)).sum(dim=-1)
    if active is not None:
        ent = ent * active.to(ent.dtype)
    return log_probs, ent


def _masked_smoothed_nll(
    log_probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact label-smoothed NLL restricted strictly to legal (masked-in) options.

    For each active row with K legal options:
      loss = -(1 - eps) * log_p[target] - eps * (sum_{legal j} log_p[j]) / K
    When K == 1 (forced move), both terms are 0. Inactive rows return exact 0.
    """
    B = log_probs.shape[0]
    ar = torch.arange(B, device=log_probs.device)
    nll_hard = -log_probs[ar, target]
    if label_smoothing > 0.0:
        legal_f = mask.to(log_probs.dtype)
        legal_cnt = legal_f.sum(dim=-1).clamp(min=1.0)
        mean_neg_lp = -(log_probs.clamp(min=-1e4) * legal_f).sum(dim=-1) / legal_cnt
        loss = (1.0 - label_smoothing) * nll_hard + label_smoothing * mean_neg_lp
    else:
        loss = nll_hard
    if active is not None:
        loss = loss * active.to(loss.dtype)
    return loss


def _sample_batch_masked(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    deterministic: bool = False,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized batched masked sampling over `[B, K]` logits -> `(idx [B], log_prob [B], entropy [B])`.

    Returned `log_prob` and `entropy` always come from the unscaled `T=1.0` distribution so PPO
    importance ratios remain exact; inactive rows return `log_prob=0.0`, `entropy=0.0`.
    """
    log_probs, entropy = _masked_log_softmax_and_entropy(logits, mask, active=active)
    safe_mask = mask.clone()
    if active is not None:
        inactive = ~active
        if inactive.any():
            safe_mask[inactive, 0] = True

    if deterministic or temperature <= 0.0:
        idx = logits.masked_fill(~safe_mask, -1e9).argmax(dim=-1)
    elif abs(temperature - 1.0) < 1e-6:
        probs = log_probs.exp()
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        scaled = (logits / float(temperature)).masked_fill(~safe_mask, -1e9)
        sample_probs = F.softmax(scaled, dim=-1)
        idx = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)

    ar = torch.arange(logits.shape[0], device=logits.device)
    chosen_lp = log_probs[ar, idx]
    if active is not None:
        chosen_lp = chosen_lp * active.to(chosen_lp.dtype)
    return idx, chosen_lp, entropy


class StrongholdPolicyNet(nn.Module):
    """Tokenized Transformer backbone with decoupled Policy/Value readouts & 4-step Pointer Head."""

    def __init__(self, catalog: GameCatalog | None = None, config: PolicyConfig | None = None) -> None:
        super().__init__()
        cat = catalog if catalog is not None else get_catalog()
        cfg = config if config is not None else PolicyConfig()
        self.cfg = cfg
        self.num_entities = cat.num_entities
        self.num_items = cat.num_items
        self.num_bonds = cat.num_bonds
        self.num_bands = cat.num_bands
        self.num_stages = cat.num_stages
        d = cfg.d_model

        # ---- Isolated categorical embedding tables (0 = PAD/NONE) ----------
        self.seg_emb = nn.Embedding(NUM_SEGMENTS, d)
        self.slot_emb = nn.Embedding(NUM_TOKENS, d)
        self.entity_emb = nn.Embedding(self.num_entities, d, padding_idx=0)
        self.equip_emb = nn.Embedding(self.num_items, d, padding_idx=0)
        self.equip_slot_emb = nn.Embedding(MAX_EQUIP_SLOTS, d)

        self.band_emb = nn.Embedding(self.num_bands + 1, d, padding_idx=0)
        self.bond_emb = nn.Embedding(self.num_bonds + 1, d, padding_idx=0)
        self.stage_emb = nn.Embedding(self.num_stages, d, padding_idx=0)
        self.dir_emb = nn.Embedding(5, d, padding_idx=0)

        # Static entity -> bond prior projection (zero-shot bond synergy awareness)
        bond_table = torch.zeros((self.num_entities, self.num_bonds), dtype=torch.float32)
        for cm in cat.chess_meta.values():
            for bi in cm.bond_indices:
                bond_table[cm.entity_idx, bi] = 1.0
        self.register_buffer("entity_bond_table", bond_table, persistent=False)
        self.bond_prior_proj = nn.Linear(self.num_bonds, d, bias=False)

        self.cont_proj = nn.Sequential(
            nn.Linear(CONT_DIM, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(cfg.dropout)

        # ---- Transformer encoder -------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.out_drop = nn.Dropout(cfg.dropout)

        # ---- Decoupled Critic Value Head (reads VALUE_CLS_IDX) -------------
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # ---- Step 1: Action Type (reads POLICY_CLS_IDX) --------------------
        self.type_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, NUM_ACTION_TYPES),
        )
        self.type_emb = nn.Embedding(NUM_ACTION_TYPES, d)

        # ---- Step 2: Source Entity (Pointer Attention) + Band Head ---------
        self.src_ctx_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.src_key_proj = nn.Linear(d, d, bias=False)
        self.band_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.num_bands))

        # ---- Step 3: Target Slot/Entity (Pointer Attention) ----------------
        self.tgt_ctx_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.tgt_key_proj = nn.Linear(d, d, bias=False)

        # ---- Step 4: Sub-option (Board Facing 0..3 / Equip Replace 0..5) ---
        self.opt_ctx_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.dir_head = nn.Linear(d, NUM_SUB_OPTS)
        self.replace_head = nn.Linear(d, NUM_SUB_OPTS)

        self._scale = 1.0 / math.sqrt(d)

    def encode_tokens(
        self,
        cat_ids: torch.Tensor,    # [B, N, NUM_CAT_FIELDS]
        cont_feats: torch.Tensor, # [B, N, CONT_DIM]
        valid_mask: torch.Tensor, # [B, N] bool
    ) -> torch.Tensor:
        """Run isolated embeddings + Transformer encoder -> [B, N, d_model]."""
        B, N, _ = cat_ids.shape
        device = cat_ids.device
        slots = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        seg = self.seg_emb(cat_ids[:, :, CAT_SEG])
        pos = self.slot_emb(slots)
        ent = self.entity_emb(cat_ids[:, :, CAT_ENTITY])
        eq_ids = cat_ids[:, :, CAT_EQUIP0 : CAT_EQUIP0 + MAX_EQUIP_SLOTS]
        eq_valid = (eq_ids > 0).unsqueeze(-1)
        eq = ((self.equip_emb(eq_ids) + self.equip_slot_emb.weight.view(1, 1, MAX_EQUIP_SLOTS, -1)) * eq_valid).sum(dim=2)
        band = self.band_emb(cat_ids[:, :, CAT_BAND])
        bond = self.bond_emb(cat_ids[:, :, CAT_BOND])
        stage = self.stage_emb(cat_ids[:, :, CAT_STAGE])
        direction = self.dir_emb(cat_ids[:, :, CAT_DIR])
        bond_prior = self.bond_prior_proj(self.entity_bond_table[cat_ids[:, :, CAT_ENTITY]])
        cont = self.cont_proj(cont_feats)

        x = self.in_drop(self.in_norm(seg + pos + ent + eq + band + bond + stage + direction + bond_prior + cont))
        h = self.encoder(x, src_key_padding_mask=~valid_mask)
        return self.out_drop(self.out_norm(h))

    @torch.no_grad()
    def act_batch_tensors(
        self,
        cat_ids: torch.Tensor,
        cont_feats: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_specs: Sequence[ActionMaskSpec | ActionTree],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Fully vectorized batched Transformer forward + 4-step autoregressive pointer decoding.

        Accepts pre-stacked CPU tensors `[B, N, ...]` and lightweight `ActionMaskSpec` (or `ActionTree`)
        objects, performing at most 4 bulk GPU->CPU syncs per batch regardless of batch size `B`.
        """
        B = int(cat_ids.shape[0])
        if B == 0:
            return {
                "a_type": [],
                "a_src": [],
                "a_tgt": [],
                "a_opt": [],
                "log_prob": [],
                "entropy": [],
                "value": [],
                "type_mask": torch.zeros((0, NUM_ACTION_TYPES), dtype=torch.bool),
                "src_mask": torch.zeros((0, NUM_TOKENS), dtype=torch.bool),
                "has_src": torch.zeros((0,), dtype=torch.bool),
                "band_mask": torch.zeros((0, self.num_bands), dtype=torch.bool),
                "has_band": torch.zeros((0,), dtype=torch.bool),
                "tgt_mask": torch.zeros((0, NUM_TOKENS), dtype=torch.bool),
                "has_tgt": torch.zeros((0,), dtype=torch.bool),
                "opt_mask": torch.zeros((0, NUM_SUB_OPTS), dtype=torch.bool),
                "has_opt": torch.zeros((0,), dtype=torch.bool),
            }

        cat_dev = cat_ids.to(device, non_blocking=True)
        cont_dev = cont_feats.to(device, non_blocking=True)
        valid_dev = valid_mask.to(device, non_blocking=True)
        ar = torch.arange(B, device=device)

        h_all = self.encode_tokens(cat_dev, cont_dev, valid_dev)    # [B, N, D]
        g_pol_all = h_all[:, POLICY_CLS_IDX]                        # [B, D]
        g_val_all = h_all[:, VALUE_CLS_IDX]                         # [B, D]
        values_all = self.value_head(g_val_all).squeeze(-1)         # [B]
        type_logits_all = self.type_head(g_pol_all)                 # [B, NUM_ACTION_TYPES]

        # ---- Step 1: Action Type -------------------------------------------
        type_mask_cpu = torch.zeros((B, NUM_ACTION_TYPES), dtype=torch.bool)
        for i, spec in enumerate(mask_specs):
            if isinstance(spec, ActionMaskSpec):
                tm = np.frombuffer(spec.type_mask_bytes, dtype=np.bool_)
                type_mask_cpu[i] = torch.from_numpy(tm.copy())
            else:
                type_mask_cpu[i] = spec.type_mask

        a_type_dev, total_lp, total_ent = _sample_batch_masked(
            type_logits_all,
            type_mask_cpu.to(device, non_blocking=True),
            deterministic=deterministic,
            temperature=temperature,
        )
        a_type_list: list[int] = a_type_dev.cpu().tolist()

        # ---- Step 2: Source Entity Pointer or Band Choice ------------------
        src_mask_cpu = torch.zeros((B, NUM_TOKENS), dtype=torch.bool)
        has_src_cpu = torch.zeros(B, dtype=torch.bool)
        band_mask_cpu = torch.zeros((B, self.num_bands), dtype=torch.bool)
        has_band_cpu = torch.zeros(B, dtype=torch.bool)
        any_src = False
        any_band = False

        for i, (spec, a_type) in enumerate(zip(mask_specs, a_type_list)):
            srcs = spec.src_keys.get(a_type)
            if not srcs:
                continue
            if a_type == TYPE_BAND:
                band_mask_cpu[i, srcs] = True
                has_band_cpu[i] = True
                any_band = True
            else:
                src_mask_cpu[i, srcs] = True
                has_src_cpu[i] = True
                any_src = True

        a_src_list = [NO_ARG] * B
        a_tgt_list = [NO_ARG] * B
        a_opt_list = [NO_ARG] * B

        tgt_mask_cpu = torch.zeros((B, NUM_TOKENS), dtype=torch.bool)
        has_tgt_cpu = torch.zeros(B, dtype=torch.bool)
        opt_mask_cpu = torch.zeros((B, NUM_SUB_OPTS), dtype=torch.bool)
        has_opt_cpu = torch.zeros(B, dtype=torch.bool)

        if any_src or any_band:
            ctx1 = self.src_ctx_proj(torch.cat([g_pol_all, self.type_emb(a_type_dev)], dim=-1))
            src_choice_list: list[int] | None = None
            band_choice_list: list[int] | None = None

            if any_src:
                src_keys_all = self.src_key_proj(h_all)
                src_logits = torch.einsum("bd,bnd->bn", ctx1, src_keys_all) * self._scale
                src_idx, lp_src, ent_src = _sample_batch_masked(
                    src_logits,
                    src_mask_cpu.to(device, non_blocking=True),
                    active=has_src_cpu.to(device, non_blocking=True),
                    deterministic=deterministic,
                    temperature=temperature,
                )
                total_lp = total_lp + lp_src
                total_ent = total_ent + ent_src
                src_choice_list = src_idx.cpu().tolist()

            if any_band:
                band_logits = self.band_head(ctx1)
                band_idx, lp_band, ent_band = _sample_batch_masked(
                    band_logits,
                    band_mask_cpu.to(device, non_blocking=True),
                    active=has_band_cpu.to(device, non_blocking=True),
                    deterministic=deterministic,
                    temperature=temperature,
                )
                total_lp = total_lp + lp_band
                total_ent = total_ent + ent_band
                band_choice_list = band_idx.cpu().tolist()

            any_tgt = False
            for i in range(B):
                if has_band_cpu[i]:
                    assert band_choice_list is not None
                    a_src_list[i] = band_choice_list[i]
                elif has_src_cpu[i]:
                    assert src_choice_list is not None
                    s_idx = src_choice_list[i]
                    a_src_list[i] = s_idx
                    tgts = mask_specs[i].tgt_keys.get((a_type_list[i], s_idx))
                    if tgts:
                        tgt_mask_cpu[i, tgts] = True
                        has_tgt_cpu[i] = True
                        any_tgt = True

            # ---- Step 3: Target Pointer ------------------------------------
            if any_tgt:
                safe_src = torch.tensor(
                    [s if 0 <= s < NUM_TOKENS else 0 for s in a_src_list],
                    dtype=torch.long,
                    device=device,
                )
                src_h = h_all[ar, safe_src]
                ctx2 = self.tgt_ctx_proj(torch.cat([ctx1, src_h], dim=-1))
                tgt_keys_all = self.tgt_key_proj(h_all)
                tgt_logits = torch.einsum("bd,bnd->bn", ctx2, tgt_keys_all) * self._scale
                tgt_idx, lp_tgt, ent_tgt = _sample_batch_masked(
                    tgt_logits,
                    tgt_mask_cpu.to(device, non_blocking=True),
                    active=has_tgt_cpu.to(device, non_blocking=True),
                    deterministic=deterministic,
                    temperature=temperature,
                )
                total_lp = total_lp + lp_tgt
                total_ent = total_ent + ent_tgt
                tgt_choice_list: list[int] = tgt_idx.cpu().tolist()

                any_opt = False
                for i in range(B):
                    if not has_tgt_cpu[i]:
                        continue
                    t_idx = tgt_choice_list[i]
                    a_tgt_list[i] = t_idx
                    spec = mask_specs[i]
                    key3 = (a_type_list[i], a_src_list[i], t_idx)
                    if isinstance(spec, ActionMaskSpec):
                        bits = spec.opt_bits.get(key3, 0)
                        if bits:
                            for o in range(NUM_SUB_OPTS):
                                if bits & (1 << o):
                                    opt_mask_cpu[i, o] = True
                            has_opt_cpu[i] = True
                            any_opt = True
                    else:
                        opts = spec.opt_keys.get(key3)
                        if opts:
                            opt_mask_cpu[i, opts] = True
                            has_opt_cpu[i] = True
                            any_opt = True

                # ---- Step 4: Sub-option (Direction / Replace Slot) ---------
                if any_opt:
                    safe_tgt = torch.tensor(
                        [t if 0 <= t < NUM_TOKENS else 0 for t in a_tgt_list],
                        dtype=torch.long,
                        device=device,
                    )
                    tgt_h = h_all[ar, safe_tgt]
                    ctx3 = self.opt_ctx_proj(torch.cat([ctx2, tgt_h], dim=-1))
                    is_equip = (a_type_dev == TYPE_EQUIP).unsqueeze(-1)
                    opt_logits = torch.where(is_equip, self.replace_head(ctx3), self.dir_head(ctx3))
                    opt_idx, lp_opt, ent_opt = _sample_batch_masked(
                        opt_logits,
                        opt_mask_cpu.to(device, non_blocking=True),
                        active=has_opt_cpu.to(device, non_blocking=True),
                        deterministic=deterministic,
                        temperature=temperature,
                    )
                    total_lp = total_lp + lp_opt
                    total_ent = total_ent + ent_opt
                    opt_choice_list: list[int] = opt_idx.cpu().tolist()
                    for i in range(B):
                        if has_opt_cpu[i]:
                            a_opt_list[i] = opt_choice_list[i]

        scalars = torch.stack([total_lp, total_ent, values_all], dim=0).cpu().tolist()
        return {
            "a_type": a_type_list,
            "a_src": a_src_list,
            "a_tgt": a_tgt_list,
            "a_opt": a_opt_list,
            "log_prob": scalars[0],
            "entropy": scalars[1],
            "value": scalars[2],
            "type_mask": type_mask_cpu,
            "src_mask": src_mask_cpu,
            "has_src": has_src_cpu,
            "band_mask": band_mask_cpu,
            "has_band": has_band_cpu,
            "tgt_mask": tgt_mask_cpu,
            "has_tgt": has_tgt_cpu,
            "opt_mask": opt_mask_cpu,
            "has_opt": has_opt_cpu,
        }

    @torch.no_grad()
    def act_batch(
        self,
        encs: Sequence[EncodedFrame],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> list[dict[str, Any]]:
        """Batched Transformer forward pass + vectorized 4-step autoregressive decoding."""
        if not encs:
            return []
        cat_ids = torch.stack([e.cat_ids for e in encs])
        cont = torch.stack([e.cont_feats for e in encs])
        valid = torch.stack([e.valid_mask for e in encs])
        trees = [e.action_tree for e in encs]

        out = self.act_batch_tensors(
            cat_ids,
            cont,
            valid,
            trees,
            deterministic=deterministic,
            temperature=temperature,
            device=device,
        )

        results: list[dict[str, Any]] = []
        for i, tree in enumerate(trees):
            a_type = out["a_type"][i]
            a_src = out["a_src"][i]
            a_tgt = out["a_tgt"][i]
            a_opt = out["a_opt"][i]
            action_key = tree.decode(a_type, a_src, a_tgt, a_opt)
            masks = StepMasks(
                type_mask=out["type_mask"][i],
                src_mask=out["src_mask"][i],
                has_src=bool(out["has_src"][i].item()),
                band_mask=out["band_mask"][i],
                has_band=bool(out["has_band"][i].item()),
                tgt_mask=out["tgt_mask"][i],
                has_tgt=bool(out["has_tgt"][i].item()),
                opt_mask=out["opt_mask"][i],
                has_opt=bool(out["has_opt"][i].item()),
            )
            results.append(
                {
                    "action_key": action_key,
                    "action_tuple": (a_type, a_src, a_tgt, a_opt),
                    "log_prob": float(out["log_prob"][i]),
                    "entropy": float(out["entropy"][i]),
                    "value": float(out["value"][i]),
                    "masks": masks,
                }
            )
        return results

    @torch.no_grad()
    def act_single(
        self,
        enc: EncodedFrame,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Autoregressively sample one legal composite action for a single EncodedFrame."""
        return self.act_batch([enc], deterministic=deterministic, temperature=temperature, device=device)[0]

    def band_log_probs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate only the opening BAND distribution, using the rollout's legal masks."""
        h = self.encode_tokens(batch["cat_ids"], batch["cont_feats"], batch["valid_mask"])
        g_pol = h[:, POLICY_CLS_IDX]
        a_type = torch.full((h.shape[0],), TYPE_BAND, dtype=torch.long, device=h.device)
        ctx = self.src_ctx_proj(torch.cat([g_pol, self.type_emb(a_type)], dim=-1))
        log_probs, _ = _masked_log_softmax_and_entropy(self.band_head(ctx), batch["band_mask"])
        return log_probs

    def evaluate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        label_smoothing: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Fully vectorized evaluation of a batch of composite actions for BC & Masked PPO."""
        cat_ids = batch["cat_ids"]         # [B, N, 8]
        cont_feats = batch["cont_feats"]   # [B, N, CONT_DIM]
        valid_mask = batch["valid_mask"]   # [B, N]
        a_type = batch["a_type"]           # [B]
        a_src = batch["a_src"]             # [B]
        a_tgt = batch["a_tgt"]             # [B]
        a_opt = batch["a_opt"]             # [B]

        type_mask = batch["type_mask"]     # [B, NUM_ACTION_TYPES]
        src_mask = batch["src_mask"]       # [B, N]
        has_src = batch["has_src"]         # [B]
        band_mask = batch["band_mask"]     # [B, NUM_BANDS]
        has_band = batch["has_band"]       # [B]
        tgt_mask = batch["tgt_mask"]       # [B, N]
        has_tgt = batch["has_tgt"]         # [B]
        opt_mask = batch["opt_mask"]       # [B, NUM_SUB_OPTS]
        has_opt = batch["has_opt"]         # [B]

        B = cat_ids.shape[0]
        ar = torch.arange(B, device=cat_ids.device)

        h = self.encode_tokens(cat_ids, cont_feats, valid_mask)  # [B, N, D]
        g_pol = h[:, POLICY_CLS_IDX]                             # [B, D]
        g_val = h[:, VALUE_CLS_IDX]                              # [B, D]
        values = self.value_head(g_val).squeeze(-1)              # [B]

        # Step 1: Action Type
        type_logits = self.type_head(g_pol)
        type_lp_all, type_ent = _masked_log_softmax_and_entropy(type_logits, type_mask)
        lp_type = type_lp_all[ar, a_type]

        # Step 2: Source Entity Pointer + Band Choice
        ctx1 = self.src_ctx_proj(torch.cat([g_pol, self.type_emb(a_type)], dim=-1))  # [B, D]

        src_keys = self.src_key_proj(h)  # [B, N, D]
        src_logits = torch.einsum("bd,bnd->bn", ctx1, src_keys) * self._scale
        src_lp_all, src_ent = _masked_log_softmax_and_entropy(src_logits, src_mask, active=has_src)
        safe_src = a_src.clamp(min=0, max=NUM_TOKENS - 1)
        lp_src = src_lp_all[ar, safe_src] * has_src.to(lp_type.dtype)

        band_logits = self.band_head(ctx1)
        band_lp_all, band_ent = _masked_log_softmax_and_entropy(band_logits, band_mask, active=has_band)
        safe_band = a_src.clamp(min=0, max=self.num_bands - 1)
        lp_band = band_lp_all[ar, safe_band] * has_band.to(lp_type.dtype)

        # Step 3: Target Pointer
        src_h = h[ar, safe_src]
        ctx2 = self.tgt_ctx_proj(torch.cat([ctx1, src_h], dim=-1))
        tgt_keys = self.tgt_key_proj(h)
        tgt_logits = torch.einsum("bd,bnd->bn", ctx2, tgt_keys) * self._scale
        tgt_lp_all, tgt_ent = _masked_log_softmax_and_entropy(tgt_logits, tgt_mask, active=has_tgt)
        safe_tgt = a_tgt.clamp(min=0, max=NUM_TOKENS - 1)
        lp_tgt = tgt_lp_all[ar, safe_tgt] * has_tgt.to(lp_type.dtype)

        # Step 4: Sub-option (Direction for MOVE, Replace Slot for EQUIP)
        tgt_h = h[ar, safe_tgt]
        ctx3 = self.opt_ctx_proj(torch.cat([ctx2, tgt_h], dim=-1))
        dir_logits = self.dir_head(ctx3)
        rep_logits = self.replace_head(ctx3)
        is_equip = (a_type == TYPE_EQUIP).unsqueeze(-1)
        opt_logits = torch.where(is_equip, rep_logits, dir_logits)
        opt_lp_all, opt_ent = _masked_log_softmax_and_entropy(opt_logits, opt_mask, active=has_opt)
        safe_opt = a_opt.clamp(min=0, max=NUM_SUB_OPTS - 1)
        lp_opt = opt_lp_all[ar, safe_opt] * has_opt.to(lp_type.dtype)

        joint_lp = lp_type + lp_src + lp_band + lp_tgt + lp_opt
        joint_ent = type_ent + src_ent + band_ent + tgt_ent + opt_ent

        if label_smoothing > 0.0:
            ls_type = _masked_smoothed_nll(type_lp_all, a_type, type_mask, label_smoothing)
            ls_src = _masked_smoothed_nll(src_lp_all, safe_src, src_mask, label_smoothing, active=has_src)
            ls_band = _masked_smoothed_nll(band_lp_all, safe_band, band_mask, label_smoothing, active=has_band)
            ls_tgt = _masked_smoothed_nll(tgt_lp_all, safe_tgt, tgt_mask, label_smoothing, active=has_tgt)
            ls_opt = _masked_smoothed_nll(opt_lp_all, safe_opt, opt_mask, label_smoothing, active=has_opt)
            bc_loss = ls_type + ls_src + ls_band + ls_tgt + ls_opt
        else:
            bc_loss = -joint_lp

        pred_type = type_logits.masked_fill(~type_mask, -1e9).argmax(dim=-1)
        pred_src = src_logits.masked_fill(~src_mask, -1e9).argmax(dim=-1)
        pred_band = band_logits.masked_fill(~band_mask, -1e9).argmax(dim=-1)
        pred_tgt = tgt_logits.masked_fill(~tgt_mask, -1e9).argmax(dim=-1)
        pred_opt = opt_logits.masked_fill(~opt_mask, -1e9).argmax(dim=-1)

        return {
            "log_prob": joint_lp,
            "bc_loss": bc_loss,
            "entropy": joint_ent,
            "type_ent": type_ent,
            "src_ent": src_ent,
            "band_ent": band_ent,
            "tgt_ent": tgt_ent,
            "opt_ent": opt_ent,
            "value": values,
            "lp_type": lp_type,
            "lp_src": lp_src,
            "lp_band": lp_band,
            "lp_tgt": lp_tgt,
            "lp_opt": lp_opt,
            "pred_type": pred_type,
            "pred_src": pred_src,
            "pred_band": pred_band,
            "pred_tgt": pred_tgt,
            "pred_opt": pred_opt,
        }
