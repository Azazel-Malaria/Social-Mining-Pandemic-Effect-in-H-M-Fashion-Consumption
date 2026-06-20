from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[0]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.social_behavior_model import SocialBehaviorModel
from util.hm_behavior_dataset import HMBehaviorDataset, make_behavior_collate_fn
from util.io_utils import ensure_dir, format_float_dict, move_to_device, resolve_device, save_json, seed_everything
from util.losses import multi_positive_softmax_loss, soft_target_cross_entropy, selected_prompt_infonce_loss
from util.metrics import MetricAverager, topk_metrics
from util.style_taxonomy import STYLES


def parse_bool_arg(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def setup_wandb(args, model_dir: Path):
    if not bool(args.use_wandb):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("--use_wandb 1 was set, but wandb is not installed. Run `pip install wandb`.") from exc
    tags = [t.strip() for t in str(args.wandb_tags or "").split(",") if t.strip()]
    run_name = args.wandb_run_name or f"clip_transformer_{'K' if args.use_knowledge else 'noK'}_seed{args.seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.wandb_group or ("clip_transformer_K" if args.use_knowledge else "clip_transformer_noK"),
        tags=tags,
        mode=args.wandb_mode,
        config=vars(args),
        dir=str(model_dir),
    )


def parameter_report(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modules = []
    for name, module in model.named_children():
        mt = sum(p.numel() for p in module.parameters())
        mr = sum(p.numel() for p in module.parameters() if p.requires_grad)
        modules.append({
            "module": name,
            "total_params": int(mt),
            "trainable_params": int(mr),
            "frozen_params": int(mt - mr),
            "trainable_ratio": float(mr / max(mt, 1)),
        })
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "trainable_ratio": float(trainable / max(total, 1)),
        "modules": modules,
    }


def print_parameter_report(report: dict):
    print("========== Stage-2 trainable parameter report ==========")
    print(f"Total params:     {report['total_params']:,}")
    print(f"Trainable params: {report['trainable_params']:,} ({report['trainable_ratio'] * 100:.2f}%)")
    print(f"Frozen params:    {report['frozen_params']:,}")
    for row in report["modules"]:
        status = "trainable" if row["trainable_params"] == row["total_params"] else ("frozen" if row["trainable_params"] == 0 else "partial")
        print(f"  - {row['module']:<22} {status:<9} trainable={row['trainable_params']:,} / total={row['total_params']:,}")
    print("=======================================================")


def compute_losses(out: dict, batch: dict, args) -> dict:
    loss_7day = multi_positive_softmax_loss(out["scores"], batch["labels"], batch["candidate_mask"])
    loss_style = soft_target_cross_entropy(out["style_logits"], batch["candidate_style"], batch["candidate_mask"])
    zero = out["scores"].sum() * 0.0

    # Prompt-level InfoNCE: selected routed prompt is positive; unselected
    # prompts under the same subject/dimension are negatives.  This is not the
    # recommendation negative-item loss.
    loss_prompt = zero
    if bool(args.use_knowledge) and args.lambda_prompt_infonce > 0 and "prompt_pos_emb" in out:
        # prompt_item_emb contains only items selected for prompt InfoNCE
        # (default: positive purchased candidates).  This avoids materializing
        # prompt positives/negatives for all recommendation negative items.
        loss_prompt = selected_prompt_infonce_loss(
            item_emb=out["prompt_item_emb"],
            pos_prompt_emb=out["prompt_pos_emb"],
            neg_prompt_emb=out["prompt_neg_emb"],
            prompt_mask=batch.get("prompt_mask"),
            item_mask=batch.get("prompt_item_valid"),
            neg_mask=batch.get("prompt_neg_mask"),
            temperature=args.prompt_temperature,
        )

    loss_anchor = out.get("anchor_loss", zero)
    total = (
        loss_7day
        + args.lambda_style * loss_style
        + args.lambda_prompt_infonce * loss_prompt
        + args.lambda_anchor * loss_anchor
    )
    return {
        "loss": total,
        "loss_7day": loss_7day,
        "loss_style": loss_style,
        "loss_prompt_infonce": loss_prompt,
        "loss_anchor": loss_anchor,
        "gate_mean": out.get("gate_mean", zero),
    }

def run_epoch(model, loader, optimizer, device, args, split: str, train: bool, epoch: int, wandb_run=None, scaler=None) -> Dict[str, float]:
    model.train(train)
    avg = MetricAverager()
    pbar = tqdm(loader, desc=f"{split} epoch", leave=False)
    for step_idx, batch in enumerate(pbar, start=1):
        batch = move_to_device(batch, device)
        use_amp = bool(args.amp) and str(device).startswith("cuda")
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(batch)
                losses = compute_losses(out, batch, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    scaler.scale(losses["loss"]).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
        bsz = int(batch["labels"].size(0))
        loss_metrics = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        rank_metrics = topk_metrics(out["scores"], batch["labels"], batch["candidate_mask"], k=args.topk_metric)
        avg.update({**loss_metrics, **rank_metrics}, n=bsz)
        if wandb_run is not None and train and args.wandb_log_train_steps and (step_idx % args.wandb_log_interval == 0):
            global_step = (epoch - 1) * max(len(loader), 1) + step_idx
            wandb_run.log({f"train_step/{k}": v for k, v in {**loss_metrics, **rank_metrics}.items()}, step=global_step)
        pbar.set_postfix({"loss": f"{loss_metrics['loss']:.3f}", f"hit@{args.topk_metric}": f"{rank_metrics[f'hit_rate@{args.topk_metric}']:.3f}"})
    return avg.compute()


def save_checkpoint(path: Path, model, optimizer, args, epoch: int, best_metric: float):
    ensure_dir(path.parent)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--output_root", default="./output")
    p.add_argument("--item_prefix", default="clip")
    p.add_argument("--use_knowledge", type=int, default=0, choices=[0, 1])
    p.add_argument("--include_temporal", type=int, default=0, choices=[0, 1])
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1], help="Use CUDA autocast mixed precision to reduce activation memory.")
    p.add_argument("--num_negatives", type=int, default=None, help="If omitted, use all non-positive/non-history items as negatives. Set e.g. 32 for sampled negatives.")
    p.add_argument("--max_history_items", type=int, default=None, help="If omitted, keep full user history.")
    p.add_argument("--max_train_samples", type=int, default=None, help="If omitted, use all train windows.")
    p.add_argument("--max_eval_samples", type=int, default=None, help="If omitted, use all val/test windows.")
    p.add_argument("--sample_strategy", choices=["random", "stratified_month", "stratified_month_group"], default="stratified_month_group")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--item_adapter_layers", type=int, default=2)
    p.add_argument("--item_adapter_heads", type=int, default=4)
    p.add_argument("--user_layers", type=int, default=2)
    p.add_argument("--user_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lambda_style", type=float, default=0.05)
    p.add_argument("--lambda_prompt_infonce", type=float, default=0.05)
    p.add_argument("--prompt_temperature", type=float, default=0.07)
    p.add_argument("--prompt_negatives_per_positive", type=int, default=2)
    p.add_argument("--prompt_negative_mode", choices=["bottomk", "all_unselected"], default="bottomk", help="Prompt InfoNCE negatives: bottomk uses least-similar prompts stored during routing; all_unselected is a legacy fallback.")
    p.add_argument("--prompt_infonce_on", choices=["positives", "candidates"], default="positives")
    p.add_argument("--prompt_infonce_max_items_per_batch", type=int, default=None, help="Optional cap for prompt InfoNCE items per batch. Omit for no cap.")
    p.add_argument("--lambda_anchor", type=float, default=0.01)
    p.add_argument("--use_two_tower", type=int, default=0, choices=[0, 1], help="0=ordinary candidate-history Transformer; 1=fast two-tower dot-product scoring.")
    p.add_argument("--item_encode_chunk_size", type=int, default=2048, help="Micro-batch item+knowledge encoding to bound peak memory.")
    p.add_argument("--candidate_chunk_size", type=int, default=4, help="For use_two_tower=0, score candidates in chunks to avoid B*C*H expansion.")
    p.add_argument("--interaction_layers", type=int, default=None)
    p.add_argument("--interaction_heads", type=int, default=None)
    p.add_argument("--transformer_injection", type=parse_bool_arg, default=False, help="If true, inject compact knowledge factors into candidate-history Transformer layers. Default false keeps the old path unchanged.")
    p.add_argument("--transformer_injection_layers", type=int, default=0, help="How many interaction Transformer layers receive knowledge-factor modulation. 0 means all layers.")
    p.add_argument("--transformer_injection_strength", type=float, default=1.0, help="Residual strength alpha for knowledge-factor modulation.")
    p.add_argument("--topk_metric", type=int, default=12)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="HM_social_mining")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--wandb_tags", default="two_stage,clip_transformer")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb_log_train_steps", type=int, default=0)
    p.add_argument("--wandb_log_interval", type=int, default=50)
    args = p.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.cuda_id)
    print(f"[train_behavior] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}; resolved device={device}")
    model_name = "clip_transformer_k" if args.use_knowledge else "clip_transformer_nok"
    model_dir = ensure_dir(Path(args.output_root) / model_name)
    save_json(vars(args), model_dir / "config.json")

    ds_kwargs = dict(
        data_root=args.data_root,
        item_prefix=args.item_prefix,
        use_knowledge=bool(args.use_knowledge),
        include_temporal=bool(args.include_temporal),
        num_negatives=args.num_negatives,
        max_history_items=args.max_history_items,
        sample_strategy=args.sample_strategy,
        prompt_negatives_per_positive=args.prompt_negatives_per_positive,
        prompt_negative_mode=args.prompt_negative_mode,
        prompt_infonce_on=args.prompt_infonce_on,
        prompt_infonce_max_items_per_batch=args.prompt_infonce_max_items_per_batch,
    )
    train_ds = HMBehaviorDataset(split="train", seed=args.seed, max_samples=args.max_train_samples, **ds_kwargs)
    val_ds = HMBehaviorDataset(split="val", seed=args.seed + 1, max_samples=args.max_eval_samples, **ds_kwargs)
    test_ds = HMBehaviorDataset(split="test", seed=args.seed + 2, max_samples=args.max_eval_samples, **ds_kwargs)
    # Store the inferred factor dimension in args/checkpoints so social inference
    # can reconstruct models trained with --transformer_injection True.
    args.knowledge_factor_dim = int(getattr(train_ds, "knowledge_factor_dim", 0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available(), collate_fn=make_behavior_collate_fn(train_ds))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(), collate_fn=make_behavior_collate_fn(val_ds))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available(), collate_fn=make_behavior_collate_fn(test_ds))

    model = SocialBehaviorModel(
        base_dim=train_ds.base_dim,
        hidden_dim=args.hidden_dim,
        num_styles=len(STYLES),
        use_knowledge=bool(args.use_knowledge),
        knowledge_dim=train_ds.knowledge_dim if args.use_knowledge else 0,
        item_adapter_layers=args.item_adapter_layers,
        item_adapter_heads=args.item_adapter_heads,
        user_layers=args.user_layers,
        user_heads=args.user_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_history=args.max_history_items or max(train_ds.max_observed_history_len, 1),
        use_two_tower=bool(args.use_two_tower),
        interaction_layers=args.interaction_layers,
        interaction_heads=args.interaction_heads,
        item_encode_chunk_size=args.item_encode_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
        transformer_injection=bool(args.transformer_injection),
        knowledge_factor_dim=args.knowledge_factor_dim,
        transformer_injection_layers=args.transformer_injection_layers,
        transformer_injection_strength=args.transformer_injection_strength,
    ).to(device)

    report = parameter_report(model)
    print_parameter_report(report)
    save_json(report, model_dir / "trainable_params.json")

    wandb_run = setup_wandb(args, model_dir)
    if wandb_run is not None:
        wandb_run.summary.update({k: report[k] for k in ["total_params", "trainable_params", "frozen_params", "trainable_ratio"]})

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and str(device).startswith("cuda"))
    rows = []
    best_metric = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, "train", True, epoch, wandb_run, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, "val", False, epoch, wandb_run, scaler=None)
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            rows.append({"epoch": epoch, "split": split, **format_float_dict(metrics, 3)})
        pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)
        target_metric = val_metrics.get(f"ndcg@{args.topk_metric}", 0.0)
        if wandb_run is not None:
            payload = {"epoch": epoch}
            payload.update({f"train/{k}": float(v) for k, v in train_metrics.items()})
            payload.update({f"val/{k}": float(v) for k, v in val_metrics.items()})
            wandb_run.log(payload, step=epoch)
        save_checkpoint(model_dir / "last_model.pt", model, optimizer, args, epoch, best_metric)
        if target_metric > best_metric:
            best_metric = target_metric
            best_epoch = epoch
            save_checkpoint(model_dir / "best_model.pt", model, optimizer, args, epoch, best_metric)
        print(f"Epoch {epoch}: val_ndcg@{args.topk_metric}={target_metric:.3f}, best={best_metric:.3f} at epoch {best_epoch}")

    best_path = model_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    test_metrics = run_epoch(model, test_loader, optimizer, device, args, "test", False, best_epoch, wandb_run, scaler=None)
    rows.append({"epoch": best_epoch, "split": "test", **format_float_dict(test_metrics, 3)})
    pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": float(v) for k, v in test_metrics.items()}, step=args.epochs + 1)
        wandb_run.summary[f"best_val_ndcg@{args.topk_metric}"] = best_metric
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
    print(f"Training finished. Metrics saved to {model_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
