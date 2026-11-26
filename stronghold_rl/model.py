"""Transformer encoder (114 semantic tokens, isolated embeddings) + 4-step Autoregressive Pointer Head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

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
    CAT_EQUIP1,
    CAT_SEG,
    CAT_STAGE,
    CONT_DIM,
    NO_ARG,
    NUM_ACTION_TYPES,
    NUM_SEGMENTS,
    NUM_SUB_OPTS,
    NUM_TOKENS,
    POLICY_CLS_IDX,
    TYPE_BAND,
    TYPE_EQUIP,
    VALUE_CLS_IDX,
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


def _sample_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    deterministic: bool,
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Sample or argmax one index from a 1D masked logit vector, returning (idx, log_prob, entropy).

    Note: returned `log_prob` and `entropy` are always from the unscaled policy distribution (`T=1.0`)
    so PPO importance ratios remain exact; `temperature` only shapes sampling sharpness during inference.
    """
    masked = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.clamp(min=-1e4)).sum()
    if deterministic or temperature <= 0.0:
        idx = int(torch.argmax(masked).item())
    elif abs(temperature - 1.0) < 1e-6:
        idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        scaled = (logits / float(temperature)).masked_fill(~mask, -1e9)
        sample_probs = F.softmax(scaled, dim=-1)
        idx = int(torch.multinomial(sample_probs, num_samples=1).item())
    return idx, log_probs[idx], entropy


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
        self.equip_emb = nn.Embedding(self.num_items, d // 2, padding_idx=0)
        self.equip_proj = nn.Linear(d, d, bias=False)

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

        # ---- Step 4: Sub-option (Board Facing 0..3 / Equip Replace 0..3) ---
        self.opt_ctx_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.dir_head = nn.Linear(d, NUM_SUB_OPTS)
        self.replace_head = nn.Linear(d, NUM_SUB_OPTS)

        self._scale = 1.0 / math.sqrt(d)

    def encode_tokens(
        self,
        cat_ids: torch.Tensor,    # [B, N, 8]
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
        eq = self.equip_proj(
            torch.cat(
                [self.equip_emb(cat_ids[:, :, CAT_EQUIP0]), self.equip_emb(cat_ids[:, :, CAT_EQUIP1])],
                dim=-1,
            )
        )
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
    def act_batch(
        self,
        encs: Sequence[EncodedFrame],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> list[dict[str, Any]]:
        """Batched Transformer forward pass + per-item 4-step autoregressive decoding."""
        if not encs:
            return []
        cat_ids = torch.stack([e.cat_ids for e in encs]).to(device)
        cont = torch.stack([e.cont_feats for e in encs]).to(device)
        valid = torch.stack([e.valid_mask for e in encs]).to(device)

        h_all = self.encode_tokens(cat_ids, cont, valid)    # [B, N, D]
        g_pol_all = h_all[:, POLICY_CLS_IDX]                # [B, D]
        g_val_all = h_all[:, VALUE_CLS_IDX]                 # [B, D]
        values_all = self.value_head(g_val_all).squeeze(-1) # [B]
        type_logits_all = self.type_head(g_pol_all)         # [B, NUM_ACTION_TYPES]
        src_keys_all = self.src_key_proj(h_all)             # [B, N, D]
        tgt_keys_all = self.tgt_key_proj(h_all)             # [B, N, D]

        results: list[dict[str, Any]] = []
        for i, enc in enumerate(encs):
            h = h_all[i]
            g_pol = g_pol_all[i]
            tree = enc.action_tree
            total_lp = torch.zeros((), device=device)
            total_ent = torch.zeros((), device=device)

            # Step 1: Action Type
            type_mask = tree.type_mask.to(device)
            a_type, lp1, ent1 = _sample_from_logits(type_logits_all[i], type_mask, deterministic, temperature)
            total_lp = total_lp + lp1
            total_ent = total_ent + ent1

            a_src = NO_ARG
            a_tgt = NO_ARG
            a_opt = NO_ARG

            # Step 2: Source Entity Pointer or Band Choice
            sm = tree.src_mask(a_type)
            if sm is not None:
                t_emb = self.type_emb(torch.tensor(a_type, device=device))
                ctx1 = self.src_ctx_proj(torch.cat([g_pol, t_emb], dim=-1))
                sm_dev = sm.to(device)
                if a_type == TYPE_BAND:
                    band_logits = self.band_head(ctx1)
                    a_src, lp2, ent2 = _sample_from_logits(band_logits, sm_dev, deterministic, temperature)
                else:
                    src_logits = (src_keys_all[i] @ ctx1) * self._scale
                    a_src, lp2, ent2 = _sample_from_logits(src_logits, sm_dev, deterministic, temperature)
                total_lp = total_lp + lp2
                total_ent = total_ent + ent2

                # Step 3: Target Pointer
                tm = tree.tgt_mask(a_type, a_src)
                if tm is not None:
                    src_h = h[a_src]
                    ctx2 = self.tgt_ctx_proj(torch.cat([ctx1, src_h], dim=-1))
                    tgt_logits = (tgt_keys_all[i] @ ctx2) * self._scale
                    a_tgt, lp3, ent3 = _sample_from_logits(tgt_logits, tm.to(device), deterministic, temperature)
                    total_lp = total_lp + lp3
                    total_ent = total_ent + ent3

                    # Step 4: Sub-option (Direction or Replace Equip Slot)
                    om = tree.opt_mask(a_type, a_src, a_tgt)
                    if om is not None:
                        tgt_h = h[a_tgt]
                        ctx3 = self.opt_ctx_proj(torch.cat([ctx2, tgt_h], dim=-1))
                        opt_logits = self.replace_head(ctx3) if a_type == TYPE_EQUIP else self.dir_head(ctx3)
                        a_opt, lp4, ent4 = _sample_from_logits(opt_logits, om.to(device), deterministic, temperature)
                        total_lp = total_lp + lp4
                        total_ent = total_ent + ent4

            action_key = tree.decode(a_type, a_src, a_tgt, a_opt)
            masks = extract_step_masks(enc, a_type, a_src, a_tgt, a_opt)
            results.append(
                {
                    "action_key": action_key,
                    "action_tuple": (a_type, a_src, a_tgt, a_opt),
                    "log_prob": float(total_lp.item()),
                    "entropy": float(total_ent.item()),
                    "value": float(values_all[i].item()),
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
