from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from model.common import (
    AdaptiveImageTextFusion,
    CrossAttentionKnowledgeSelector,
    GatedResidualFusion,
    MLPProjection,
    MeanUserEncoder,
    StyleHead,
    dot_scores,
)


class ClipMultimodalModel(nn.Module):
    """CLIP image-text model with Amazon-Qwen knowledge injection.

    H&M article text anchors the CLIP text side; Amazon-Qwen knowledge tokens
    are injected into the image side through attention selection and gated
    residual fusion. The final item representation is a learned mixture of the
    knowledge-enhanced visual embedding and the H&M article-text embedding.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14",
                 freeze_clip: bool = True, knowledge_dim: int = 256,
                 embed_dim: int = 256, num_styles: int = 10,
                 num_query_tokens: int = 8, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as exc:
            raise ImportError("Please install transformers to use ClipMultimodalModel.") from exc
        self.clip_model_name = clip_model_name
        self.freeze_clip = bool(freeze_clip)
        self.knowledge_dim = knowledge_dim
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        if self.freeze_clip:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad = False
        feature_dim = int(getattr(self.clip.config, "projection_dim", embed_dim))
        self.clip_feature_dim = feature_dim
        self.image_projector = MLPProjection(feature_dim, embed_dim, dropout)
        self.text_projector = MLPProjection(feature_dim, embed_dim, dropout)
        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.selector = CrossAttentionKnowledgeSelector(embed_dim, num_query_tokens, num_heads, dropout)
        self.knowledge_fusion = GatedResidualFusion(embed_dim, dropout)
        self.image_text_fusion = AdaptiveImageTextFusion(embed_dim, dropout)
        self.user_encoder = MeanUserEncoder(embed_dim)
        self.style_head = StyleHead(embed_dim, num_styles)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def _clip_features(self, images: torch.Tensor, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if self.freeze_clip else nullcontext()
        with ctx:
            image_feat = self.clip.get_image_features(pixel_values=images)
            text_feat = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return image_feat.float(), text_feat.float()

    def encode_items(self, images: torch.Tensor, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor, knowledge_tokens: torch.Tensor,
                     reliability: torch.Tensor | None = None,
                     knowledge_mask: torch.Tensor | None = None) -> dict:
        shape = images.shape[:-3]
        flat_images = images.reshape(-1, *images.shape[-3:])
        flat_ids = input_ids.reshape(-1, input_ids.size(-1))
        flat_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        kt = knowledge_tokens.reshape(-1, knowledge_tokens.size(-2), knowledge_tokens.size(-1))
        rel = reliability.reshape(-1) if reliability is not None else None
        km = knowledge_mask.reshape(-1, knowledge_mask.size(-1)) if knowledge_mask is not None else None

        image_feat, text_feat = self._clip_features(flat_images, flat_ids, flat_mask)
        image_emb = self.image_projector(image_feat)
        text_emb = self.text_projector(text_feat)
        kt = self.knowledge_projector(kt)
        knowledge_emb, attn_weights, query_pool = self.selector(image_emb, kt, km)
        visual_knowledge_emb, gate, residual = self.knowledge_fusion(image_emb, knowledge_emb, rel)
        fused_emb, image_text_lambda = self.image_text_fusion(visual_knowledge_emb, text_emb)

        residual_norm = residual.pow(2).sum(dim=-1)
        if rel is not None:
            reliability_gap = (1.0 - rel).clamp_min(0.0)
            budget_loss = (reliability_gap * residual_norm).mean()
            anchor_loss = (reliability_gap * (visual_knowledge_emb - image_emb).pow(2).sum(dim=-1)).mean()
        else:
            budget_loss = residual_norm.mean()
            anchor_loss = (visual_knowledge_emb - image_emb).pow(2).sum(dim=-1).mean()

        return {
            "image_emb": image_emb.reshape(*shape, -1),
            "text_emb": text_emb.reshape(*shape, -1),
            "knowledge_emb": knowledge_emb.reshape(*shape, -1),
            "visual_knowledge_emb": visual_knowledge_emb.reshape(*shape, -1),
            "fused_emb": fused_emb.reshape(*shape, -1),
            "gate": gate.reshape(*shape, -1),
            "image_text_lambda": image_text_lambda.reshape(*shape),
            "budget_loss": budget_loss,
            "anchor_loss": anchor_loss,
            "attn_weights": attn_weights,
            "query_pool": query_pool,
        }

    def forward(self, batch: dict) -> dict:
        h = self.encode_items(
            batch["history_images"], batch["history_input_ids"], batch["history_attention_mask"],
            batch["history_knowledge"], batch["history_reliability"],
        )
        c = self.encode_items(
            batch["candidate_images"], batch["candidate_input_ids"], batch["candidate_attention_mask"],
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
            "candidate_visual_emb": c["image_emb"],
            "candidate_knowledge_emb": c["knowledge_emb"],
            "candidate_clip_image_emb": c["visual_knowledge_emb"],
            "candidate_text_emb": c["text_emb"],
            "history_clip_image_emb": h["visual_knowledge_emb"],
            "history_text_emb": h["text_emb"],
            "style_logits": style_logits,
            "budget_loss": budget_loss,
            "anchor_loss": anchor_loss,
            "gate": c["gate"],
            "image_text_lambda": c["image_text_lambda"],
        }
