from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def dot_scores(user_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,bnd->bn", F.normalize(user_emb, dim=-1), F.normalize(cand_emb, dim=-1))


class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CrossKnowledgeInjectionLayer(nn.Module):
    """Layerwise prompt injection for a single item token.

    The routed knowledge is already compact: each item only carries top-1/top-2
    prompts per dimension selected offline by image/text similarity.  This layer
    performs a small cross-attention from the item token to those prompts and a
    gated residual update.  It is intentionally not a large multimodal backbone.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, item: torch.Tensor, knowledge: torch.Tensor, knowledge_mask: torch.Tensor | None = None):
        # item: [N, D], knowledge: [N, M, D], knowledge_mask: [N, M]
        if knowledge_mask is not None:
            valid = knowledge_mask.any(dim=1)
            key_padding_mask = ~knowledge_mask.bool()
        else:
            valid = torch.ones(item.size(0), dtype=torch.bool, device=item.device)
            key_padding_mask = None
        q = item.unsqueeze(1)
        attn_out, attn_weights = self.attn(q, knowledge, knowledge, key_padding_mask=key_padding_mask, need_weights=True)
        attn_out = attn_out.squeeze(1)
        attn_out = torch.where(valid.unsqueeze(-1), attn_out, torch.zeros_like(attn_out))
        gate = self.gate(torch.cat([item, attn_out], dim=-1))
        x = self.norm1(item + gate * attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, gate, attn_weights.squeeze(1)


class ItemKnowledgeAdapter(nn.Module):
    """Trainable item adapter.

    Without knowledge it is a trainable projection from frozen CLIP-like item
    features to behavior space.  With knowledge it additionally performs
    layerwise prompt injection; no image/text/LLM backbone is trained here.
    """

    def __init__(self, base_dim: int, knowledge_dim: int, hidden_dim: int = 256, layers: int = 2,
                 num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.base_projector = MLPProjection(base_dim, hidden_dim, dropout)
        self.knowledge_projector = MLPProjection(knowledge_dim, hidden_dim, dropout) if knowledge_dim > 0 else None
        self.layers = nn.ModuleList([
            CrossKnowledgeInjectionLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(max(int(layers), 0))
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def project_prompts(self, prompt_features: torch.Tensor) -> torch.Tensor:
        if self.knowledge_projector is None:
            raise RuntimeError("project_prompts requires knowledge_projector, but knowledge_dim was 0.")
        shp = prompt_features.shape[:-1]
        x = prompt_features.reshape(-1, prompt_features.size(-1)).float()
        y = self.knowledge_projector(x)
        return y.reshape(*shp, -1)

    def forward(self, base_features: torch.Tensor, knowledge: torch.Tensor | None = None,
                knowledge_mask: torch.Tensor | None = None) -> dict:
        shape = base_features.shape[:-1]
        x0 = base_features.reshape(-1, base_features.size(-1)).float()
        base_emb = self.base_projector(x0)
        if knowledge is None or self.knowledge_projector is None or len(self.layers) == 0:
            emb = self.out_norm(base_emb)
            return {
                "item_emb": emb.reshape(*shape, -1),
                "base_emb": base_emb.reshape(*shape, -1),
                "knowledge_emb": torch.zeros_like(base_emb).reshape(*shape, -1),
                "anchor_loss": base_emb.sum() * 0.0,
                "gate_mean": base_emb.sum() * 0.0,
            }

        k = knowledge.reshape(-1, knowledge.size(-2), knowledge.size(-1)).float()
        km = knowledge_mask.reshape(-1, knowledge_mask.size(-1)).bool() if knowledge_mask is not None else None
        k_proj = self.knowledge_projector(k)
        x = base_emb
        gates = []
        attn_last = None
        for layer in self.layers:
            x, gate, attn_last = layer(x, k_proj, km)
            gates.append(gate)
        x = self.out_norm(x)
        if km is not None:
            m = km.to(dtype=k_proj.dtype).unsqueeze(-1)
            k_mean = (k_proj * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            k_mean = k_proj.mean(dim=1)
        anchor_loss = (1.0 - F.cosine_similarity(x, base_emb.detach(), dim=-1)).mean()
        gate_mean = torch.stack([g.mean() for g in gates]).mean() if gates else x.sum() * 0.0
        return {
            "item_emb": x.reshape(*shape, -1),
            "base_emb": base_emb.reshape(*shape, -1),
            "knowledge_emb": k_mean.reshape(*shape, -1),
            "anchor_loss": anchor_loss,
            "gate_mean": gate_mean,
            "attention_weights": attn_last,
        }


class KnowledgeFactorModulation(nn.Module):
    """Low-memory layerwise knowledge-factor modulation.

    Instead of attending to raw prompt tokens inside the candidate-history
    Transformer, this module injects compact, interpretable item factors such as
    routed prompt scores and weak style scores.  It implements

        x' = x + strength * sigmoid(W_g[x; r]) * r,
        r  = W_a a,

    where a is a per-item factor vector.  This keeps the item-level social
    variables stable while still allowing every selected Transformer layer to be
    conditioned on knowledge.
    """

    def __init__(self, factor_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.factor_proj = nn.Sequential(
            nn.Linear(self.factor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, factors: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: [N,S,D], factors: [N,S,F]
        r = self.factor_proj(factors.float()).to(dtype=hidden.dtype)
        gate = self.gate(torch.cat([hidden, r], dim=-1))
        out = self.norm(hidden + float(strength) * gate * r)
        return out, gate


class HistoryTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, layers: int = 2, num_heads: int = 4,
                 ffn_dim: int = 512, dropout: float = 0.1, max_history: int = 80):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.randn(1, max_history + 1, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hist_emb: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        b, h, _ = hist_emb.shape
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, hist_emb], dim=1)
        if x.size(1) > self.pos.size(1):
            pos = F.interpolate(self.pos.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos[:, :x.size(1)]
        x = x + pos
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=hist_mask.device)
        mask = torch.cat([cls_mask, hist_mask.bool()], dim=1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(x[:, 0])


class CandidateInteractionTransformer(nn.Module):
    """Ordinary Transformer scoring path for 7-day prediction.

    Default path is intentionally unchanged: when transformer_injection=False,
    this class uses nn.TransformerEncoder exactly as before.

    Optional path: when transformer_injection=True, each selected layer first
    applies compact knowledge-factor modulation to the candidate/history tokens:

        X_l <- X_l + alpha * gate(X_l, A) * W_a A,
        X_{l+1} <- TransformerLayer_l(X_l),

    where A contains stable per-item knowledge/style factors.  This realizes
    history/candidate layerwise knowledge injection without making the exported
    social item variables context-dependent.
    """

    def __init__(self, hidden_dim: int = 256, layers: int = 2, num_heads: int = 4,
                 ffn_dim: int = 512, dropout: float = 0.1, max_history: int = 80,
                 candidate_chunk_size: int = 4,
                 transformer_injection: bool = False,
                 knowledge_factor_dim: int = 0,
                 transformer_injection_layers: int | None = None,
                 transformer_injection_strength: float = 1.0):
        super().__init__()
        self.candidate_chunk_size = max(1, int(candidate_chunk_size)) if candidate_chunk_size is not None else 4
        self.num_layers = max(1, int(layers))
        self.transformer_injection = bool(transformer_injection) and int(knowledge_factor_dim) > 0
        self.knowledge_factor_dim = int(knowledge_factor_dim)
        self.injection_strength = float(transformer_injection_strength)
        if transformer_injection_layers is None or int(transformer_injection_layers) <= 0:
            self.injection_layers = self.num_layers
        else:
            self.injection_layers = min(int(transformer_injection_layers), self.num_layers)
        self.type_emb = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)  # 0=candidate, 1=history
        self.pos = nn.Parameter(torch.randn(1, max_history + 1, hidden_dim) * 0.02)

        if self.transformer_injection:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ])
            self.factor_mod = KnowledgeFactorModulation(self.knowledge_factor_dim, hidden_dim, dropout)
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
            self.factor_mod = None
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def _encode_sequence(self, seq: torch.Tensor, valid: torch.Tensor, factors: torch.Tensor | None = None) -> torch.Tensor:
        if not self.transformer_injection:
            return self.encoder(seq, src_key_padding_mask=~valid.bool())
        if factors is None:
            factors = torch.zeros(seq.size(0), seq.size(1), self.knowledge_factor_dim, device=seq.device, dtype=seq.dtype)
        x = seq
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.injection_layers:
                x, _ = self.factor_mod(x, factors, strength=self.injection_strength)
            x = layer(x, src_key_padding_mask=~valid.bool())
        return x

    def _score_chunk(self, hist_emb: torch.Tensor, hist_mask: torch.Tensor,
                     cand_emb: torch.Tensor, cand_mask: torch.Tensor,
                     hist_factors: torch.Tensor | None = None,
                     cand_factors: torch.Tensor | None = None) -> torch.Tensor:
        b, h, d = hist_emb.shape
        c = cand_emb.size(1)
        cand_flat = cand_emb.reshape(b * c, d)
        # expand only this chunk, not the full candidate set
        hist_rep = hist_emb.unsqueeze(1).expand(b, c, h, d).reshape(b * c, h, d)
        seq = torch.cat([cand_flat.unsqueeze(1), hist_rep], dim=1)
        cand_valid = cand_mask.reshape(b * c, 1)
        hist_valid = hist_mask.unsqueeze(1).expand(b, c, h).reshape(b * c, h)
        valid = torch.cat([cand_valid, hist_valid], dim=1)

        type_add = torch.cat([
            self.type_emb[:, :1].expand(b * c, 1, d),
            self.type_emb[:, 1:2].expand(b * c, h, d),
        ], dim=1)
        if seq.size(1) > self.pos.size(1):
            pos = F.interpolate(self.pos.transpose(1, 2), size=seq.size(1), mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos[:, :seq.size(1)]
        seq = seq + type_add + pos

        factor_seq = None
        if self.transformer_injection:
            fdim = self.knowledge_factor_dim
            if hist_factors is None:
                hist_factors = torch.zeros(b, h, fdim, device=seq.device, dtype=seq.dtype)
            if cand_factors is None:
                cand_factors = torch.zeros(b, c, fdim, device=seq.device, dtype=seq.dtype)
            cand_f = cand_factors.reshape(b * c, fdim)
            hist_f = hist_factors.unsqueeze(1).expand(b, c, h, fdim).reshape(b * c, h, fdim)
            factor_seq = torch.cat([cand_f.unsqueeze(1), hist_f], dim=1)
            # zero out padding factors to avoid injecting into masked history/candidate tokens
            factor_seq = factor_seq * valid.to(dtype=factor_seq.dtype).unsqueeze(-1)

        out = self._encode_sequence(seq, valid, factor_seq)
        score = self.scorer(self.norm(out[:, 0])).reshape(b, c).float()
        # AMP may produce fp16/bf16 scores.  Filling fp16 with -1e9 overflows,
        # so keep logits in fp32 before masking.
        return score.masked_fill(~cand_mask.bool(), -1e9)

    def forward(self, hist_emb: torch.Tensor, hist_mask: torch.Tensor,
                cand_emb: torch.Tensor, cand_mask: torch.Tensor,
                hist_factors: torch.Tensor | None = None,
                cand_factors: torch.Tensor | None = None) -> torch.Tensor:
        c_total = cand_emb.size(1)
        if c_total <= self.candidate_chunk_size:
            return self._score_chunk(hist_emb, hist_mask, cand_emb, cand_mask, hist_factors, cand_factors)
        chunks = []
        for st in range(0, c_total, self.candidate_chunk_size):
            ed = min(st + self.candidate_chunk_size, c_total)
            cf = cand_factors[:, st:ed] if cand_factors is not None else None
            chunks.append(self._score_chunk(hist_emb, hist_mask, cand_emb[:, st:ed], cand_mask[:, st:ed], hist_factors, cf))
        return torch.cat(chunks, dim=1)

class SocialBehaviorModel(nn.Module):
    """Stage-2 behavior/social model.

    Two final model groups remain:
      use_knowledge=False: frozen CLIP-like item features + trainable Transformer.
      use_knowledge=True: the same, plus layerwise routed-prompt injection.

    Scoring has two modes:
      use_two_tower=True: user tower and item tower dot product, faster.
      use_two_tower=False: ordinary candidate-history interaction Transformer.
    """

    def __init__(self, base_dim: int, hidden_dim: int = 256, num_styles: int = 10,
                 use_knowledge: bool = False, knowledge_dim: int = 0,
                 use_two_tower: bool = False,
                 item_adapter_layers: int = 2, item_adapter_heads: int = 4,
                 user_layers: int = 2, user_heads: int = 4,
                 interaction_layers: int | None = None, interaction_heads: int | None = None,
                 ffn_dim: int = 512, dropout: float = 0.1, max_history: int = 80,
                 item_encode_chunk_size: int = 2048, candidate_chunk_size: int = 4,
                 transformer_injection: bool = False,
                 knowledge_factor_dim: int = 0,
                 transformer_injection_layers: int | None = None,
                 transformer_injection_strength: float = 1.0):
        super().__init__()
        self.use_knowledge = bool(use_knowledge)
        self.use_two_tower = bool(use_two_tower)
        self.transformer_injection = bool(transformer_injection) and self.use_knowledge and (not self.use_two_tower) and int(knowledge_factor_dim) > 0
        self.knowledge_factor_dim = int(knowledge_factor_dim) if self.transformer_injection else 0
        self.item_encode_chunk_size = max(1, int(item_encode_chunk_size)) if item_encode_chunk_size is not None else 2048
        kdim = int(knowledge_dim) if self.use_knowledge else 0
        self.item_encoder = ItemKnowledgeAdapter(
            base_dim=base_dim,
            knowledge_dim=kdim,
            hidden_dim=hidden_dim,
            layers=item_adapter_layers if self.use_knowledge else 0,
            num_heads=item_adapter_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.user_encoder = HistoryTransformerEncoder(hidden_dim, user_layers, user_heads, ffn_dim, dropout, max_history)
        self.interaction_encoder = CandidateInteractionTransformer(
            hidden_dim=hidden_dim,
            layers=interaction_layers if interaction_layers is not None else user_layers,
            num_heads=interaction_heads if interaction_heads is not None else user_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_history=max_history,
            candidate_chunk_size=candidate_chunk_size,
            transformer_injection=self.transformer_injection,
            knowledge_factor_dim=self.knowledge_factor_dim,
            transformer_injection_layers=transformer_injection_layers,
            transformer_injection_strength=transformer_injection_strength,
        )
        self.style_head = nn.Linear(hidden_dim, num_styles)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode_items(self, base_features: torch.Tensor, knowledge: torch.Tensor | None = None,
                     knowledge_mask: torch.Tensor | None = None) -> dict:
        """Encode item tensors in chunks to bound peak memory.

        base_features may be [B,H,D] or [B,C,D].  The old implementation
        flattened everything at once, so [B,C,M,D] routed knowledge could easily
        OOM before scoring.  This function preserves the original shape but
        micro-batches the flattened item axis.
        """
        shape = base_features.shape[:-1]
        flat_base = base_features.reshape(-1, base_features.size(-1))
        if knowledge is not None:
            flat_k = knowledge.reshape(-1, knowledge.size(-2), knowledge.size(-1))
            flat_m = knowledge_mask.reshape(-1, knowledge_mask.size(-1)) if knowledge_mask is not None else None
        else:
            flat_k = None
            flat_m = None
        n = flat_base.size(0)
        if n <= self.item_encode_chunk_size:
            out = self.item_encoder(flat_base, flat_k if self.use_knowledge else None, flat_m if self.use_knowledge else None)
            return {k: (v.reshape(*shape, *v.shape[1:]) if torch.is_tensor(v) and v.dim() >= 2 and v.size(0) == n else v) for k, v in out.items()}

        parts: dict[str, list[torch.Tensor]] = {}
        anchor_vals = []
        gate_vals = []
        for st in range(0, n, self.item_encode_chunk_size):
            ed = min(st + self.item_encode_chunk_size, n)
            out = self.item_encoder(
                flat_base[st:ed],
                flat_k[st:ed] if (self.use_knowledge and flat_k is not None) else None,
                flat_m[st:ed] if (self.use_knowledge and flat_m is not None) else None,
            )
            for key in ["item_emb", "base_emb", "knowledge_emb"]:
                parts.setdefault(key, []).append(out[key])
            anchor_vals.append(out.get("anchor_loss", flat_base.sum() * 0.0) * (ed - st))
            gate_vals.append(out.get("gate_mean", flat_base.sum() * 0.0) * (ed - st))
        ret = {key: torch.cat(vals, dim=0).reshape(*shape, -1) for key, vals in parts.items()}
        ret["anchor_loss"] = torch.stack(anchor_vals).sum() / max(n, 1)
        ret["gate_mean"] = torch.stack(gate_vals).sum() / max(n, 1)
        return ret

    def _project_prompt_batch(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None or (not self.use_knowledge):
            return None
        # Project prompt tensors in chunks over the flattened prompt axis.  This
        # prevents prompt-level InfoNCE from materializing a huge projection all at once.
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))
        if flat.size(0) <= self.item_encode_chunk_size:
            return self.item_encoder.project_prompts(x)
        outs = []
        for st in range(0, flat.size(0), self.item_encode_chunk_size):
            ed = min(st + self.item_encode_chunk_size, flat.size(0))
            outs.append(self.item_encoder.knowledge_projector(flat[st:ed].float()))
        return torch.cat(outs, dim=0).reshape(*shape, -1)

    def forward(self, batch: dict) -> dict:
        if self.use_knowledge:
            h = self.encode_items(batch["history_base_features"], batch["history_knowledge"], batch["history_knowledge_mask"])
            c = self.encode_items(batch["candidate_base_features"], batch["candidate_knowledge"], batch["candidate_knowledge_mask"])
        else:
            h = self.encode_items(batch["history_base_features"])
            c = self.encode_items(batch["candidate_base_features"])

        if self.use_two_tower:
            user = self.user_encoder(h["item_emb"], batch["history_mask"])
            scores = (dot_scores(user, c["item_emb"]) * self.logit_scale.clamp(1.0, 100.0)).float()
            # AMP may produce fp16/bf16 scores.  Filling fp16 with -1e9 overflows,
            # so keep logits in fp32 before masking.
            scores = scores.masked_fill(~batch["candidate_mask"].bool(), -1e9)
        else:
            scores = self.interaction_encoder(
                h["item_emb"],
                batch["history_mask"],
                c["item_emb"],
                batch["candidate_mask"],
                batch.get("history_knowledge_factors") if self.transformer_injection else None,
                batch.get("candidate_knowledge_factors") if self.transformer_injection else None,
            )

        out = {
            "scores": scores,
            "candidate_emb": c["item_emb"],
            "history_emb": h["item_emb"],
            "candidate_base_emb": c["base_emb"],
            "candidate_knowledge_emb": c["knowledge_emb"],
            "style_logits": self.style_head(c["item_emb"]),
            "anchor_loss": 0.5 * (h["anchor_loss"] + c["anchor_loss"]),
            "gate_mean": 0.5 * (h["gate_mean"] + c["gate_mean"]),
        }
        if self.use_knowledge and "prompt_pos_features" in batch:
            # Prompt InfoNCE is computed only for the selected prompt-items prepared
            # by the dataset (default: positive purchased candidates).  Do not project
            # prompt features for every negative item candidate.
            pos = batch["prompt_item_positions"].clamp_min(0)
            bsz, pmax = pos.shape
            gather_idx = pos.unsqueeze(-1).expand(bsz, pmax, c["item_emb"].size(-1))
            prompt_item_emb = torch.gather(c["item_emb"], 1, gather_idx)
            out["prompt_item_emb"] = prompt_item_emb
            out["prompt_pos_emb"] = self._project_prompt_batch(batch["prompt_pos_features"])
            out["prompt_neg_emb"] = self._project_prompt_batch(batch["prompt_neg_features"])
        return out
