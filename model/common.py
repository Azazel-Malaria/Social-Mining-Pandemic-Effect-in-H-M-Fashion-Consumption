from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MLPProjection(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        hidden = max(embed_dim * 2, min(1024, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MeanUserEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.empty_user = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, history_emb: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        # history_emb: [B, H, D], mask: [B, H]
        mask = history_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (history_emb * mask).sum(dim=1) / denom
        empty = history_mask.sum(dim=1) == 0
        if empty.any():
            pooled[empty] = self.empty_user
        return F.normalize(pooled, dim=-1)


class StyleHead(nn.Module):
    def __init__(self, embed_dim: int, num_styles: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_styles),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class CrossAttentionKnowledgeSelector(nn.Module):
    """Legacy selector kept for compatibility with old checkpoints."""

    def __init__(self, embed_dim: int = 256, num_query_tokens: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, embed_dim) * 0.02)
        self.visual_to_query = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_emb: torch.Tensor, knowledge_tokens: torch.Tensor,
                knowledge_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # visual_emb: [N, D], knowledge_tokens: [N, L, D]
        n, _ = visual_emb.shape
        q = self.query_tokens.unsqueeze(0).expand(n, -1, -1)
        q = q + self.visual_to_query(visual_emb).unsqueeze(1)
        key_padding_mask = None
        if knowledge_mask is not None:
            key_padding_mask = ~knowledge_mask.bool()
        selected, attn_weights = self.attn(q, knowledge_tokens, knowledge_tokens,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=True,
                                            average_attn_weights=False)
        selected = self.norm(selected)
        beta = torch.softmax(self.pool(selected).squeeze(-1), dim=-1)
        k = (selected * beta.unsqueeze(-1)).sum(dim=1)
        return F.normalize(k, dim=-1), attn_weights, beta


class GatedResidualFusion(nn.Module):
    """Legacy gated residual module kept for compatibility with old checkpoints."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, visual_emb: torch.Tensor, knowledge_emb: torch.Tensor,
                reliability: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.delta(knowledge_emb)
        gate_input = torch.cat([
            visual_emb,
            delta,
            visual_emb * delta,
            torch.abs(visual_emb - delta),
        ], dim=-1)
        gate = self.gate(gate_input)
        if reliability is None:
            reliability = torch.ones(visual_emb.shape[:-1], device=visual_emb.device, dtype=visual_emb.dtype)
        while reliability.dim() < gate.dim():
            reliability = reliability.unsqueeze(-1)
        residual = reliability * gate * delta
        fused = F.normalize(visual_emb + residual, dim=-1)
        return fused, gate, residual


class AdaptiveImageTextFusion(nn.Module):
    """Learn a per-item mixture between visual-side and text-side embeddings."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_emb: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([
            visual_emb,
            text_emb,
            visual_emb * text_emb,
            torch.abs(visual_emb - text_emb),
        ], dim=-1)
        lam = self.gate(gate_input)
        fused = F.normalize(lam * visual_emb + (1.0 - lam) * text_emb, dim=-1)
        return fused, lam.squeeze(-1)


class SimilarityTopMKnowledgeSelector(nn.Module):
    """Select a small, item-conditioned subset of knowledge prompts.

    This replaces the old design that fed every prompt into cross-attention.  It
    first compares each item's visual/text representation with all category-level
    prompt embeddings and keeps only top-m prompts per item.

    The selector is lightweight and differentiable w.r.t. projection layers, but
    the top-k indices are hard selections, which is intentional for memory.
    """

    def __init__(self, embed_dim: int = 256, top_m: int = 8,
                 visual_weight: float = 1.0, text_weight: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.top_m = int(top_m)
        self.visual_weight = float(visual_weight)
        self.text_weight = float(text_weight)
        self.visual_query = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.text_query = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.knowledge_key = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_emb: torch.Tensor, knowledge_tokens: torch.Tensor,
                text_emb: Optional[torch.Tensor] = None,
                knowledge_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # visual_emb: [N,D], text_emb: [N,D] or None, knowledge_tokens: [N,L,D]
        n, l, d = knowledge_tokens.shape
        if l == 0:
            raise ValueError("knowledge_tokens must contain at least one token")
        top_m = max(1, min(self.top_m, l))
        keys = F.normalize(self.knowledge_key(knowledge_tokens), dim=-1)
        qv = F.normalize(self.visual_query(visual_emb), dim=-1)
        scores = self.visual_weight * torch.einsum("nd,nld->nl", qv, keys)
        if text_emb is not None:
            qt = F.normalize(self.text_query(text_emb), dim=-1)
            scores = scores + self.text_weight * torch.einsum("nd,nld->nl", qt, keys)
        if knowledge_mask is not None:
            scores = scores.masked_fill(~knowledge_mask.bool(), torch.finfo(scores.dtype).min)
        top_scores, top_idx = torch.topk(scores, k=top_m, dim=-1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, d)
        selected = torch.gather(knowledge_tokens, dim=1, index=gather_idx)
        selected = self.dropout(selected)
        selected_mean = F.normalize(selected.mean(dim=1), dim=-1)
        return selected, selected_mean, top_idx, top_scores


class LightweightModalityAdapter(nn.Module):
    """A small Transformer adapter for image/text/knowledge token interaction.

    Backbones remain precomputed/frozen.  This module receives only a few tokens:
    an item token, image token, optional text token, and top-m knowledge tokens.
    It returns the adjusted item token as the behavior-aligned item embedding.
    """

    def __init__(self, embed_dim: int = 256, num_layers: int = 2,
                 num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1,
                 use_text_token: bool = False):
        super().__init__()
        self.use_text_token = bool(use_text_token)
        self.item_seed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_item = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.type_image = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.type_text = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.type_knowledge = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.residual_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, base_item_emb: torch.Tensor, image_emb: torch.Tensor,
                selected_knowledge: torch.Tensor, text_emb: Optional[torch.Tensor] = None,
                reliability: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # base_item_emb/image_emb/text_emb: [N,D], selected_knowledge: [N,M,D]
        n, _, d = selected_knowledge.shape
        item_token = base_item_emb.unsqueeze(1) + self.item_seed.expand(n, -1, -1) + self.type_item
        image_token = image_emb.unsqueeze(1) + self.type_image
        tokens = [item_token, image_token]
        if self.use_text_token and text_emb is not None:
            tokens.append(text_emb.unsqueeze(1) + self.type_text)
        tokens.append(selected_knowledge + self.type_knowledge)
        x = torch.cat(tokens, dim=1)
        x = self.encoder(x)
        adapted = self.norm(x[:, 0])
        gate_input = torch.cat([
            base_item_emb,
            adapted,
            base_item_emb * adapted,
            torch.abs(base_item_emb - adapted),
        ], dim=-1)
        gate = self.residual_gate(gate_input)
        if reliability is None:
            reliability = torch.ones(base_item_emb.shape[0], device=base_item_emb.device, dtype=base_item_emb.dtype)
        while reliability.dim() < gate.dim():
            reliability = reliability.unsqueeze(-1)
        residual = reliability * gate * (adapted - base_item_emb)
        fused = F.normalize(base_item_emb + residual, dim=-1)
        return fused, gate, residual


def dot_scores(user_emb: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
    # user_emb: [B,D], candidate_emb: [B,C,D]
    return torch.einsum("bd,bcd->bc", user_emb, candidate_emb)
