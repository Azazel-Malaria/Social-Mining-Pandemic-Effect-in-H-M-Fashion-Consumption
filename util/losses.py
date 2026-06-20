from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_softmax_loss(scores: torch.Tensor,
                                target_probs: torch.Tensor,
                                candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-positive sampled softmax loss.

    scores: [B, C]
    target_probs: [B, C], non-negative and summing to 1 over positives.
    candidate_mask: [B, C], True for valid candidate positions.
    """
    # Keep logits in fp32 for numerical stability under AMP.  fp16 cannot
    # represent -1e9, so masking half-precision logits with -1e9 overflows.
    scores = scores.float()
    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask.bool(), -1e9)
        target_probs = target_probs.masked_fill(~candidate_mask.bool(), 0.0)
    log_probs = F.log_softmax(scores, dim=-1)
    per_row = -(target_probs * log_probs).sum(dim=-1)
    valid_rows = target_probs.sum(dim=-1) > 0
    if valid_rows.any():
        return per_row[valid_rows].mean()
    return scores.sum() * 0.0


def soft_target_cross_entropy(logits: torch.Tensor,
                              target_probs: torch.Tensor,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cross entropy for soft style labels.

    logits: [B, C, K] or [N, K]
    target_probs: same leading dims.
    mask: [B, C] or [N], optional.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    if mask is not None:
        loss = loss.masked_select(mask)
    if loss.numel() == 0:
        return logits.sum() * 0.0
    return loss.mean()


def symmetric_contrastive_loss(visual_emb: torch.Tensor,
                               knowledge_emb: torch.Tensor,
                               mask: torch.Tensor | None = None,
                               temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between visual and selected knowledge embeddings.

    visual_emb / knowledge_emb can be [B, C, D] or [N, D]. Mask marks valid rows.
    """
    if visual_emb.dim() == 3:
        visual_emb = visual_emb.reshape(-1, visual_emb.size(-1))
        knowledge_emb = knowledge_emb.reshape(-1, knowledge_emb.size(-1))
        if mask is not None:
            mask = mask.reshape(-1)
    if mask is not None:
        visual_emb = visual_emb[mask]
        knowledge_emb = knowledge_emb[mask]
    if visual_emb.size(0) <= 1:
        return visual_emb.sum() * 0.0
    visual_emb = F.normalize(visual_emb, dim=-1)
    knowledge_emb = F.normalize(knowledge_emb, dim=-1)
    logits = visual_emb @ knowledge_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))



def selected_prompt_infonce_loss(item_emb: torch.Tensor,
                                 pos_prompt_emb: torch.Tensor,
                                 neg_prompt_emb: torch.Tensor,
                                 prompt_mask: torch.Tensor,
                                 item_mask: torch.Tensor | None = None,
                                 neg_mask: torch.Tensor | None = None,
                                 temperature: float = 0.07) -> torch.Tensor:
    """Prompt-level InfoNCE.

    Positive prompts are the prompts selected for the current item by offline
    image/text routing.  Negative prompts are unselected prompts from the same
    subject and same knowledge dimension.  This is intentionally different from
    7-day recommendation negative items.

    item_emb: [B, C, D]
    pos_prompt_emb: [B, C, P, D]
    neg_prompt_emb: [B, C, P, N, D]
    prompt_mask: [B, C, P]
    item_mask: optional [B, C], e.g. candidate positives only.
    neg_mask: optional [B, C, P, N], True for valid bottom-k negative prompts.
    """
    if pos_prompt_emb is None or neg_prompt_emb is None:
        return item_emb.sum() * 0.0
    if neg_prompt_emb.size(-2) == 0:
        return item_emb.sum() * 0.0
    item = F.normalize(item_emb, dim=-1).unsqueeze(2)  # [B,C,1,D]
    pos = F.normalize(pos_prompt_emb, dim=-1)
    neg = F.normalize(neg_prompt_emb, dim=-1)
    pos_logit = (item * pos).sum(dim=-1, keepdim=True) / temperature  # [B,C,P,1]
    item_for_neg = item.unsqueeze(3)  # [B,C,1,1,D]
    neg_logit = (item_for_neg * neg).sum(dim=-1) / temperature  # [B,C,P,N]
    logits = torch.cat([pos_logit, neg_logit], dim=-1).float()
    valid = prompt_mask.bool()
    if neg_mask is not None:
        neg_mask = neg_mask.bool()
        logits[..., 1:] = logits[..., 1:].masked_fill(~neg_mask, -1e9)
        valid = valid & neg_mask.any(dim=-1)
    if item_mask is not None:
        valid = valid & item_mask.bool().unsqueeze(-1)
    if not valid.any():
        return item_emb.sum() * 0.0
    logits = logits[valid]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
