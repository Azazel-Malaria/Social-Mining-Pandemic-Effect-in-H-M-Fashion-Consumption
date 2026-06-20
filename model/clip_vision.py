from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from model.common import AdaptiveImageTextFusion, MLPProjection, MeanUserEncoder, StyleHead, dot_scores


class ClipVisionModel(nn.Module):
    """CLIP image-text baseline for H&M seven-day purchase supervision.

    Inputs:
      - H&M item image goes through CLIP image encoder.
      - H&M article.csv text, especially detail_desc and category fields, goes
        through CLIP text encoder.
      - The final item representation is a learned mixture of image and text.

    By default CLIP is frozen; only projection heads, image-text fusion, user
    encoder, scorer and style head are trained.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14",
                 freeze_clip: bool = True, embed_dim: int = 256,
                 num_styles: int = 10, dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as exc:
            raise ImportError("Please install transformers to use ClipVisionModel.") from exc
        self.clip_model_name = clip_model_name
        self.freeze_clip = bool(freeze_clip)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        if self.freeze_clip:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad = False
        feature_dim = int(getattr(self.clip.config, "projection_dim", embed_dim))
        self.clip_feature_dim = feature_dim
        self.image_projector = MLPProjection(feature_dim, embed_dim, dropout)
        self.text_projector = MLPProjection(feature_dim, embed_dim, dropout)
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
                     attention_mask: torch.Tensor) -> dict:
        shape = images.shape[:-3]
        flat_images = images.reshape(-1, *images.shape[-3:])
        flat_ids = input_ids.reshape(-1, input_ids.size(-1))
        flat_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        image_feat, text_feat = self._clip_features(flat_images, flat_ids, flat_mask)
        image_emb = self.image_projector(image_feat)
        text_emb = self.text_projector(text_feat)
        fused_emb, image_text_lambda = self.image_text_fusion(image_emb, text_emb)
        return {
            "image_emb": image_emb.reshape(*shape, -1),
            "text_emb": text_emb.reshape(*shape, -1),
            "fused_emb": fused_emb.reshape(*shape, -1),
            "image_text_lambda": image_text_lambda.reshape(*shape),
        }

    def forward(self, batch: dict) -> dict:
        h = self.encode_items(batch["history_images"], batch["history_input_ids"], batch["history_attention_mask"])
        c = self.encode_items(batch["candidate_images"], batch["candidate_input_ids"], batch["candidate_attention_mask"])
        user_emb = self.user_encoder(h["fused_emb"], batch["history_mask"])
        scores = dot_scores(user_emb, c["fused_emb"]) * self.logit_scale.clamp(1.0, 100.0)
        style_logits = self.style_head(c["fused_emb"])
        return {
            "scores": scores,
            "candidate_emb": c["fused_emb"],
            "history_emb": h["fused_emb"],
            "candidate_clip_image_emb": c["image_emb"],
            "candidate_text_emb": c["text_emb"],
            "history_clip_image_emb": h["image_emb"],
            "history_text_emb": h["text_emb"],
            "style_logits": style_logits,
            "image_text_lambda": c["image_text_lambda"],
        }
