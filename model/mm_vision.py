from __future__ import annotations

import torch
from torch import nn

from model.common import AdaptiveImageTextFusion, MLPProjection, MeanUserEncoder, StyleHead, dot_scores


class PrecomputedMMVisionModel(nn.Module):
    """CLIP-like dual-tower baseline using precomputed image/text features.

    This supports SigLIP/OpenCLIP features precomputed from H&M images and
    article.csv detail_desc. It does not load or run the image/text towers during
    seven-day training, so its cost is close to offline feature lookup training.
    """

    def __init__(self, image_feature_dim: int, text_feature_dim: int,
                 embed_dim: int = 256, num_styles: int = 10, dropout: float = 0.1):
        super().__init__()
        self.image_projector = MLPProjection(image_feature_dim, embed_dim, dropout)
        self.text_projector = MLPProjection(text_feature_dim, embed_dim, dropout)
        self.image_text_fusion = AdaptiveImageTextFusion(embed_dim, dropout)
        self.user_encoder = MeanUserEncoder(embed_dim)
        self.style_head = StyleHead(embed_dim, num_styles)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode_items(self, image_features: torch.Tensor, text_features: torch.Tensor) -> dict:
        image_emb = self.image_projector(image_features.float())
        text_emb = self.text_projector(text_features.float())
        fused_emb, image_text_lambda = self.image_text_fusion(image_emb, text_emb)
        return {
            "image_emb": image_emb,
            "text_emb": text_emb,
            "fused_emb": fused_emb,
            "image_text_lambda": image_text_lambda,
        }

    def forward(self, batch: dict) -> dict:
        h = self.encode_items(batch["history_mm_image_features"], batch["history_mm_text_features"])
        c = self.encode_items(batch["candidate_mm_image_features"], batch["candidate_mm_text_features"])
        user_emb = self.user_encoder(h["fused_emb"], batch["history_mask"])
        scores = dot_scores(user_emb, c["fused_emb"]) * self.logit_scale.clamp(1.0, 100.0)
        style_logits = self.style_head(c["fused_emb"])
        return {
            "scores": scores,
            "candidate_emb": c["fused_emb"],
            "history_emb": h["fused_emb"],
            "candidate_clip_image_emb": c["image_emb"],  # reused by clip alignment loss
            "candidate_text_emb": c["text_emb"],
            "history_clip_image_emb": h["image_emb"],
            "history_text_emb": h["text_emb"],
            "style_logits": style_logits,
            "image_text_lambda": c["image_text_lambda"],
        }
