from __future__ import annotations

import torch
from torch import nn

from model.common import (
    AdaptiveImageTextFusion,
    LightweightModalityAdapter,
    MLPProjection,
    MeanUserEncoder,
    SimilarityTopMKnowledgeSelector,
    StyleHead,
    dot_scores,
)


class PrecomputedMMMultimodalModel(nn.Module):
    """SigLIP/OpenCLIP-like precomputed dual-tower model with knowledge injection.

    H&M image/detail_desc features are looked up from precomputed arrays.  Amazon
    category-level Qwen prompt tokens are not fully expanded into attention;
    instead, each item selects top-m relevant prompts by visual/text similarity,
    then a small Transformer adapter fuses image, text and selected knowledge.
    """

    def __init__(self, image_feature_dim: int, text_feature_dim: int, knowledge_dim: int = 256,
                 embed_dim: int = 256, num_styles: int = 10,
                 num_query_tokens: int = 8, num_heads: int = 4, dropout: float = 0.1,
                 knowledge_top_m: int = 8, adapter_layers: int = 2,
                 adapter_num_heads: int = 4, adapter_ffn_dim: int = 512,
                 knowledge_visual_weight: float = 1.0, knowledge_text_weight: float = 1.0,
                 use_lora: bool = False, lora_rank: int = 8):
        super().__init__()
        if use_lora:
            print("[PrecomputedMMMultimodalModel] --use_lora was set, but LoRA is currently a reserved no-op; using standard lightweight adapter.")
        self.knowledge_dim = knowledge_dim
        self.image_projector = MLPProjection(image_feature_dim, embed_dim, dropout)
        self.text_projector = MLPProjection(text_feature_dim, embed_dim, dropout)
        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pre_fusion = AdaptiveImageTextFusion(embed_dim, dropout)
        self.selector = SimilarityTopMKnowledgeSelector(
            embed_dim=embed_dim,
            top_m=knowledge_top_m if knowledge_top_m is not None else num_query_tokens,
            visual_weight=knowledge_visual_weight,
            text_weight=knowledge_text_weight,
            dropout=dropout,
        )
        self.adapter = LightweightModalityAdapter(
            embed_dim=embed_dim,
            num_layers=adapter_layers,
            num_heads=adapter_num_heads,
            ffn_dim=adapter_ffn_dim,
            dropout=dropout,
            use_text_token=True,
        )
        self.user_encoder = MeanUserEncoder(embed_dim)
        self.style_head = StyleHead(embed_dim, num_styles)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode_items(self, image_features: torch.Tensor, text_features: torch.Tensor,
                     knowledge_tokens: torch.Tensor, reliability: torch.Tensor | None = None,
                     knowledge_mask: torch.Tensor | None = None) -> dict:
        shape = image_features.shape[:-1]
        img = image_features.reshape(-1, image_features.size(-1)).float()
        txt = text_features.reshape(-1, text_features.size(-1)).float()
        kt = knowledge_tokens.reshape(-1, knowledge_tokens.size(-2), knowledge_tokens.size(-1)).float()
        rel = reliability.reshape(-1) if reliability is not None else None
        km = knowledge_mask.reshape(-1, knowledge_mask.size(-1)) if knowledge_mask is not None else None

        image_emb = self.image_projector(img)
        text_emb = self.text_projector(txt)
        base_item_emb, image_text_lambda = self.pre_fusion(image_emb, text_emb)
        kt = self.knowledge_projector(kt)
        selected_knowledge, knowledge_emb, top_idx, top_scores = self.selector(image_emb, kt, text_emb, km)
        fused_emb, gate, residual = self.adapter(base_item_emb, image_emb, selected_knowledge, text_emb, rel)

        residual_norm = residual.pow(2).sum(dim=-1)
        if rel is not None:
            reliability_gap = (1.0 - rel).clamp_min(0.0)
            budget_loss = (reliability_gap * residual_norm).mean()
            anchor_loss = (reliability_gap * (fused_emb - base_item_emb).pow(2).sum(dim=-1)).mean()
        else:
            budget_loss = residual_norm.mean()
            anchor_loss = (fused_emb - base_item_emb).pow(2).sum(dim=-1).mean()

        return {
            "image_emb": image_emb.reshape(*shape, -1),
            "text_emb": text_emb.reshape(*shape, -1),
            "base_item_emb": base_item_emb.reshape(*shape, -1),
            "knowledge_emb": knowledge_emb.reshape(*shape, -1),
            "fused_emb": fused_emb.reshape(*shape, -1),
            "gate": gate.reshape(*shape, -1),
            "image_text_lambda": image_text_lambda.reshape(*shape),
            "budget_loss": budget_loss,
            "anchor_loss": anchor_loss,
            "selected_knowledge_indices": top_idx,
            "selected_knowledge_scores": top_scores,
        }

    def forward(self, batch: dict) -> dict:
        h = self.encode_items(
            batch["history_mm_image_features"], batch["history_mm_text_features"],
            batch["history_knowledge"], batch["history_reliability"],
        )
        c = self.encode_items(
            batch["candidate_mm_image_features"], batch["candidate_mm_text_features"],
            batch["candidate_knowledge"], batch["candidate_reliability"],
        )
        user_emb = self.user_encoder(h["fused_emb"], batch["history_mask"])
        scores = dot_scores(user_emb, c["fused_emb"]) * self.logit_scale.clamp(1.0, 100.0)
        style_logits = self.style_head(c["fused_emb"])
        budget_loss = 0.5 * (h["budget_loss"] + c["budget_loss"])
        anchor_loss = 0.5 * (h["anchor_loss"] + c["anchor_loss"])
        return {
            "scores": scores,
            "candidate_emb": c["fused_emb"],
            "history_emb": h["fused_emb"],
            "candidate_visual_emb": c["base_item_emb"],
            "candidate_knowledge_emb": c["knowledge_emb"],
            "candidate_clip_image_emb": c["image_emb"],
            "candidate_text_emb": c["text_emb"],
            "history_clip_image_emb": h["image_emb"],
            "history_text_emb": h["text_emb"],
            "style_logits": style_logits,
            "budget_loss": budget_loss,
            "anchor_loss": anchor_loss,
            "gate": c["gate"],
            "image_text_lambda": c["image_text_lambda"],
            "selected_knowledge_indices": c["selected_knowledge_indices"],
            "selected_knowledge_scores": c["selected_knowledge_scores"],
        }
