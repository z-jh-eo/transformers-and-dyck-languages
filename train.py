"""train.py — multitask training for DyckTransformer.

Trains detection and correction heads jointly.
Logs per-epoch metrics to a TSV and saves the best checkpoint by dev macro-F1.

Usage:
    python train.py                          # default config
    python train.py --epochs 20 --lambda-c 0.3 --smoke-test
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import make_loader, IGNORE
from model import Config, DyckTransformer


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default="train.jsonl")
    p.add_argument("--dev",        default="dev.jsonl")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--log",        default="metrics.csv")

    # training
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-c",   type=float, default=0.5,
                   help="Weight of correction loss relative to detection loss.")
    p.add_argument("--patience",   type=int,   default=3,
                   help="Early stopping patience (epochs without dev F1 improvement).")

    # model
    p.add_argument("--n-layer",  type=int,   default=4)
    p.add_argument("--n-head",   type=int,   default=4)
    p.add_argument("--d-model",  type=int,   default=128)
    p.add_argument("--dropout",  type=float, default=0.1)

    # dev
    p.add_argument("--smoke-test", action="store_true",
                   help="Overfit on 64 examples for 10 epochs to verify the pipeline.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Loss ──────────────────────────────────────────────────────────────────────

detect_criterion  = nn.CrossEntropyLoss()
correct_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

def compute_loss(detect_logits, correct_logits, detect_labels, correct_labels, lambda_c):
    """
    detect_logits:  (B, 2)
    correct_logits: (B, L, 10)
    detect_labels:  (B,)
    correct_labels: (B, L)
    """
    loss_d = detect_criterion(detect_logits, detect_labels)
    loss_c = correct_criterion(
        correct_logits.view(-1, correct_logits.size(-1)),
        correct_labels.view(-1),
    )
    return loss_d + lambda_c * loss_c, loss_d.item(), loss_c.item()


# ── Metrics ───────────────────────────────────────────────────────────────────

def detection_metrics(all_preds, all_labels):
    """Returns accuracy and macro-F1 for binary detection."""
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def correction_accuracy(all_preds, all_labels):
    """Token-level accuracy, ignoring IGNORE positions."""
    correct = total = 0
    for p, l in zip(all_preds, all_labels):
        if l != IGNORE:
            total   += 1
            correct += int(p == l)
    return correct / total if total > 0 else 0.0


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, lambda_c, train=True):
    model.train() if train else model.eval()

    total_loss = total_d = total_c = 0.0
    det_preds, det_labels   = [], []
    cor_preds, cor_labels   = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            pad_mask       = batch["pad_mask"].to(device)
            detect_lbls    = batch["detect_labels"].to(device)
            correct_lbls   = batch["correct_labels"].to(device)

            detect_logits, correct_logits = model(input_ids, pad_mask)

            loss, ld, lc = compute_loss(
                detect_logits, correct_logits,
                detect_lbls,   correct_lbls,
                lambda_c,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_d    += ld
            total_c    += lc

            det_preds.extend(detect_logits.argmax(-1).cpu().tolist())
            det_labels.extend(detect_lbls.cpu().tolist())

            cor_preds.extend(correct_logits.argmax(-1).cpu().view(-1).tolist())
            cor_labels.extend(correct_lbls.cpu().view(-1).tolist())

    n = len(loader)
    det_acc, det_f1 = detection_metrics(det_preds, det_labels)
    cor_acc          = correction_accuracy(cor_preds, cor_labels)

    return {
        "loss":    total_loss / n,
        "loss_d":  total_d    / n,
        "loss_c":  total_c    / n,
        "det_acc": det_acc,
        "det_f1":  det_f1,
        "cor_acc": cor_acc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── Loaders ──
    train_loader = make_loader(args.train, batch_size=args.batch_size, shuffle=True)
    dev_loader   = make_loader(args.dev,   batch_size=args.batch_size, shuffle=False)

    if args.smoke_test:
        # Grab one batch and overfit on it
        one_batch = [next(iter(train_loader))]
        train_loader = one_batch * 10          # pretend there are 10 "batches"
        dev_loader   = one_batch
        args.epochs  = 10
        print("smoke-test mode: overfitting on 64 examples for 10 epochs")

    # ── Model ──
    config = Config(
        n_layer  = args.n_layer,
        n_head   = args.n_head,
        d_model  = args.d_model,
        dropout  = args.dropout,
    )
    model = DyckTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Output ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log

    fieldnames = [
        "epoch",
        "train_loss", "train_loss_d", "train_loss_c",
        "train_det_acc", "train_det_f1", "train_cor_acc",
        "dev_loss",   "dev_loss_d",   "dev_loss_c",
        "dev_det_acc", "dev_det_f1",  "dev_cor_acc",
        "elapsed_s",
    ]
    log_file = open(log_path, "w", newline="")
    writer   = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    # ── Training loop ──
    best_f1      = -1.0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.lambda_c, train=True
        )
        dev_metrics = run_epoch(
            model, dev_loader,   optimizer, device, args.lambda_c, train=False
        )

        elapsed = time.time() - t0

        row = {"epoch": epoch, "elapsed_s": f"{elapsed:.1f}"}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = f"{v:.4f}"
        for k, v in dev_metrics.items():
            row[f"dev_{k}"] = f"{v:.4f}"
        writer.writerow(row)
        log_file.flush()

        print(
            f"epoch {epoch:02d} | "
            f"loss {train_metrics['loss']:.4f}/{dev_metrics['loss']:.4f} | "
            f"det F1 {train_metrics['det_f1']:.3f}/{dev_metrics['det_f1']:.3f} | "
            f"cor acc {train_metrics['cor_acc']:.3f}/{dev_metrics['cor_acc']:.3f} | "
            f"{elapsed:.0f}s"
        )

        # ── Checkpoint ──
        dev_f1 = dev_metrics["det_f1"]
        if dev_f1 > best_f1:
            best_f1      = dev_f1
            patience_ctr = 0
            ckpt = {
                "epoch":       epoch,
                "config":      config,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "dev_f1":      best_f1,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ saved best checkpoint (dev F1 = {best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  early stopping at epoch {epoch} (patience={args.patience})")
                break

    log_file.close()
    print(f"\ntraining done — best dev F1: {best_f1:.4f}")
    print(f"checkpoint: {out_dir / 'best.pt'}")
    print(f"log:        {log_path}")


if __name__ == "__main__":
    main()
