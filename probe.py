"""probe.py — Section 8: Probing Classifiers.

Q13: Global depth probe.
    Linear regression from final-layer [CLS] representation → max depth (scalar)
    Linear classification: final-layer [CLS] representation → depth class (1..7)

Q14: Local depth probe per layer.

Usage:
    python probe.py --checkpoint models/best.pt \
        --train corpus/train.jsonl \
        --test  corpus/test_id.jsonl \
        --q13
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

from data import make_loader
from model import Config, DyckTransformer

# ── CLI ───────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", required=True, help="JSONL split used to fit the probe.")
    p.add_argument(
        "--test", required=True, help="JSONL split used to evaluate the probe."
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--q13", action="store_true")
    p.add_argument("--q14", action="store_true")
    p.add_argument("--q15", action="store_true",
                   help="Q15: train probe on correct strings, evaluate on corrupted.")
    p.add_argument(
        "--min-gold-depth",
        type=int,
        default=0,
        help="Q14 diagnostic: only train/test on tokens with gold depth >= this value.",
    )
    p.add_argument(
        "--correct-only",
        action="store_true",
        help="Train and evaluate only on non-corrupted strings.",
    )
    p.add_argument(
        "--error-only",
        action="store_true",
        help="Train and evaluate only on corrupted strings.",
    )
    return p.parse_args()


# ── Representation extraction ─────────────────────────────────────────────────


def load_model(path: str, device: torch.device) -> DyckTransformer:
    ckpt = torch.load(path, map_location=device)
    config = Config(**ckpt["config"])
    model = DyckTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Local depth gold labels ───────────────────────────────────────────────────

OPENER_TO_CLOSER = {"(": ")", "[": "]"}
CLOSER_TO_OPENER = {v: k for k, v in OPENER_TO_CLOSER.items()}


def local_depth_sequence(s: str) -> list[int]:
    """Per-token local depth: stack size after consuming each token of s.
    Only meaningful for valid Dyck strings (corrupted strings give noisy values)."""
    stack = []
    depths = []
    for c in s:
        if c in OPENER_TO_CLOSER:
            stack.append(c)
        elif c in CLOSER_TO_OPENER:
            if stack and stack[-1] == CLOSER_TO_OPENER[c]:
                stack.pop()
        depths.append(len(stack))
    return depths


def _record_passes_filter(m: dict, mode: str) -> bool:
    """Return True if the record should be kept given the filter mode."""
    is_corrupted = m.get("error_type") is not None
    if mode == "correct":
        return not is_corrupted
    if mode == "error":
        return is_corrupted
    return True  # "all"


@torch.no_grad()
def extract_per_layer_representations(model, loader, device, filter_mode: str = "all"):
    """Extract per-token hidden states from every layer (including embeddings).

    `filter_mode` is one of {"all", "correct", "error"}. Gold local depth is
    computed from record["original"] (the would-be-correct string).

    Returns:
        reps   : (n_layers + 1, N_tokens, d_model) numpy array
        depths : (N_tokens,) numpy array of gold local depths
        metas  : list of per-token metadata dicts
    """
    n_layers = model.config.n_layer

    per_layer = [[] for _ in range(n_layers + 1)]
    all_depths = []
    meta_per_token = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        _, _, hidden_states, _ = model(input_ids, pad_mask, return_internals=True)
        emb = model.embed(input_ids)
        hidden_states[-1] = model.ln_f(hidden_states[-1])
        layer_outputs = [emb] + hidden_states

        for b_idx, m in enumerate(batch["meta"]):
            if not _record_passes_filter(m, filter_mode):
                continue
            original = m.get("original")
            if not original:
                continue
            gold = local_depth_sequence(original)
            n_tokens = min(len(gold), len(m["actual"]))
            for t in range(n_tokens):
                enc_pos = t + 1
                for l in range(n_layers + 1):
                    per_layer[l].append(
                        layer_outputs[l][b_idx, enc_pos, :].cpu().numpy()
                    )
                all_depths.append(gold[t])
                meta_per_token.append(
                    {
                        "id": m["id"],
                        "pos": t,
                        "is_corrupted": m.get("error_type") is not None,
                        "error_type": m.get("error_type"),
                        "error_position": m.get("error_position"),
                    }
                )

    reps = np.stack([np.stack(lst, axis=0) for lst in per_layer], axis=0)
    depths = np.array(all_depths)
    return reps, depths, meta_per_token


@torch.no_grad()
def extract_cls_representations(model, loader, device):
    """Run frozen model over a split, return final-layer CLS reps and metadata."""
    cls_reps = []
    depths = []
    metas = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        _, _, hidden_states, _ = model(input_ids, pad_mask, return_internals=True)
        final_hidden = hidden_states[-1]
        cls_vec = final_hidden[:, 0, :]

        cls_reps.append(cls_vec.cpu().numpy())
        for m in batch["meta"]:
            depths.append(m["depth"])
            metas.append(m)

    return np.concatenate(cls_reps, axis=0), np.array(depths), metas


# ── Q13: global depth probe ───────────────────────────────────────────────────


def q13(X_train, y_train, X_test, y_test, out_dir: Path, label: str):
    print(f"\n── Q13: Global depth probe  [{label}] ──────────────")

    reg = Ridge(alpha=10.0)
    reg.fit(X_train, y_train.astype(float))
    y_pred_reg = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred_reg)

    print(f"\n  Linear regression (Ridge):")
    print(f"    R² on test: {r2:.4f}")
    print(f"    train depth distribution: {dict(Counter(y_train))}")
    print(f"    test  depth distribution: {dict(Counter(y_test))}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_clf)

    print(f"\n  Linear classifier (Logistic Regression):")
    print(f"    Accuracy on test: {acc:.4f}")

    classes = sorted(set(y_train) | set(y_test))
    cm = confusion_matrix(y_test, y_pred_clf, labels=classes)

    print(f"\n  Confusion matrix (rows=true, cols=predicted, depth classes={classes}):")
    print(f"    {'true\\pred':<10}", end="")
    for c in classes:
        print(f"{c:>6}", end="")
    print()
    for i, c in enumerate(classes):
        print(f"    {c:<10}", end="")
        for v in cm[i]:
            print(f"{v:>6}", end="")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, y_pred_reg, alpha=0.3, s=10, color="steelblue")
    lo, hi = min(y_test.min(), y_pred_reg.min()), max(y_test.max(), y_pred_reg.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    axes[0].set_xlabel("true max depth")
    axes[0].set_ylabel("predicted depth (Ridge)")
    axes[0].set_title(f"Regression — R² = {r2:.3f}")
    axes[0].legend()

    im = axes[1].imshow(cm, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("predicted depth")
    axes[1].set_ylabel("true depth")
    axes[1].set_title(f"Classification — acc = {acc:.3f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle(
        f"Q13: Global depth probe on [CLS] final-layer representation  ({label})"
    )
    fig.tight_layout()
    path = out_dir / f"q13_global_depth_probe_{label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  plot saved → {path}")

    return {"r2": r2, "acc": acc}


# ── Q14: local depth probe per layer ──────────────────────────────────────────


def q14(reps_train, y_train, reps_test, y_test, out_dir: Path, label: str):
    print(f"\n── Q14: Local depth probe (per layer)  [{label}] ─────────────")

    n_layers_plus_one = reps_train.shape[0]
    r2_per_layer = []

    print(f"\n  {'Layer':<8} {'R²':>8} {'train N':>10} {'test N':>10}")
    print(f"  {'─' * 40}")

    for l in range(n_layers_plus_one):
        X_tr = reps_train[l]
        X_te = reps_test[l]
        reg = Ridge(alpha=10.0)
        reg.fit(X_tr, y_train.astype(float))
        y_pred = reg.predict(X_te)
        r2 = r2_score(y_test, y_pred)
        r2_per_layer.append(r2)

        layer_label = f"emb" if l == 0 else f"L{l}"
        print(f"  {layer_label:<8} {r2:>8.4f} {len(X_tr):>10} {len(X_te):>10}")

    best_layer = int(np.argmax(r2_per_layer))
    best_label = "emb" if best_layer == 0 else f"L{best_layer}"
    print(f"\n  Best layer: {best_label}  (R² = {r2_per_layer[best_layer]:.4f})")

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(n_layers_plus_one))
    ax.plot(xs, r2_per_layer, marker="o", color="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels(["emb"] + [f"L{l}" for l in range(1, n_layers_plus_one)])
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Probe $R^2$")
    ax.set_title(f"Q14: Local depth probe — $R^2$ per layer ({label})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    path = out_dir / f"q14_local_depth_per_layer_{label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot saved → {path}")

    return r2_per_layer


# ── Q15: error localisation via probe divergence ─────────────────────────────


@torch.no_grad()
def extract_corrupted_per_string(model, loader, device, layer_idx: int):
    """Extract per-token representations from corrupted strings only.

    Returns a list of dicts, one per corrupted record:
        {
            "reps":           (T, d_model) numpy array — representations of `actual` tokens
            "gold_depth":     (T,) list — depth in `original` for each position of `actual`
                              (NaN where there is no gold counterpart, e.g. inserted tokens)
            "error_position": int (in `actual` coordinates)
            "error_type":     str
            "id":             record id
        }

    Alignment per error type:
        E1 (deletion at p):
            actual = original[:p] + original[p+1:]
            actual[t] corresponds to original[t]       if t <  p
            actual[t] corresponds to original[t+1]     if t >= p
        E2/E4 (insertion at p):
            actual = original[:p] + new + original[p:]
            actual[t] corresponds to original[t]       if t <  p
            actual[p] is the inserted token            (no gold counterpart → NaN)
            actual[t] corresponds to original[t-1]     if t >  p
        E3 (substitution at p):
            actual = original[:p] + new + original[p+1:]
            actual[t] corresponds to original[t]       for all t
            (we still use original's gold depth at t, since the would-be-correct
             depth is what we want to compare against)
    """
    per_string = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        _, _, hidden_states, _ = model(input_ids, pad_mask, return_internals=True)
        hidden_states[-1] = model.ln_f(hidden_states[-1])
        emb = model.embed(input_ids)
        layer_outputs = [emb] + hidden_states

        H = layer_outputs[layer_idx]  # (B, L, D)

        for b_idx, m in enumerate(batch["meta"]):
            if m.get("error_type") is None:
                continue
            original = m["original"]
            actual = m["actual"]
            etype = m["error_type"]
            ep = m["error_position"]
            if original is None or actual is None or ep is None:
                continue

            gold_original = local_depth_sequence(original)

            T = len(actual)
            reps = np.empty((T, H.shape[-1]), dtype=np.float32)
            gold_aligned = []

            for t in range(T):
                enc_pos = t + 1
                reps[t] = H[b_idx, enc_pos, :].cpu().numpy()

                if etype == "e1":          # deletion at ep
                    src_idx = t if t < ep else t + 1
                    g = gold_original[src_idx] if 0 <= src_idx < len(gold_original) else float("nan")
                elif etype in ("e2", "e4"):  # insertion at ep
                    if t < ep:
                        src_idx = t
                        g = gold_original[src_idx]
                    elif t == ep:
                        g = float("nan")    # the inserted token has no gold
                    else:
                        src_idx = t - 1
                        g = gold_original[src_idx] if 0 <= src_idx < len(gold_original) else float("nan")
                elif etype == "e3":         # substitution at ep
                    src_idx = t
                    g = gold_original[src_idx] if 0 <= src_idx < len(gold_original) else float("nan")
                else:
                    g = float("nan")

                gold_aligned.append(g)

            per_string.append({
                "reps":           reps,
                "gold_depth":     np.array(gold_aligned, dtype=np.float32),
                "error_position": ep,
                "error_type":     etype,
                "id":             m["id"],
            })

    return per_string


def q15(model, train_loader, test_loader, device, out_dir: Path):
    """Train per-layer probes on CORRECT strings; apply to CORRUPTED test strings.

    Reports, for each layer:
      - R^2 on prefix tokens (before the error position)
      - R^2 on suffix tokens (at and after the error position)
      - Top-1 error localisation accuracy (position of max abs error vs gold ep)
    """
    print(f"\n── Q15: Probe divergence on corrupted strings ─────────────────")

    # ── 1. Fit probes on correct strings (training split) ─────────────────
    print("\n  Step 1/3: extracting CORRECT training representations …")
    reps_train, y_train, _ = extract_per_layer_representations(
        model, train_loader, device, filter_mode="correct"
    )
    print(f"    train shape: {reps_train.shape}")

    n_layers_plus_one = reps_train.shape[0]
    probes = []
    for l in range(n_layers_plus_one):
        reg = Ridge(alpha=10.0)
        reg.fit(reps_train[l], y_train.astype(float))
        probes.append(reg)
    print(f"  fitted {n_layers_plus_one} per-layer Ridge probes.")

    # ── 2. Apply probes to CORRUPTED test strings (per-string, aligned) ────
    print("\n  Step 2/3: extracting CORRUPTED test representations …")

    per_layer_summary = []

    for layer_idx in range(n_layers_plus_one):
        per_string = extract_corrupted_per_string(
            model, test_loader, device, layer_idx=layer_idx
        )

        # Vectorise: flatten all (token, gold) pairs across all strings,
        # masking out NaN-gold positions (inserted tokens for e2/e4).
        prefix_gold, prefix_pred = [], []
        suffix_gold, suffix_pred = [], []
        loc_hits = 0
        loc_total = 0

        for rec in per_string:
            reps = rec["reps"]
            gold = rec["gold_depth"]
            ep = rec["error_position"]

            valid_mask = ~np.isnan(gold)
            pred = probes[layer_idx].predict(reps)
            abs_err = np.abs(pred - gold)  # NaN-safe per-position

            # Split into prefix (t < ep) and suffix (t >= ep)
            T = len(gold)
            t_idx = np.arange(T)
            pre_mask = (t_idx <  ep) & valid_mask
            suf_mask = (t_idx >= ep) & valid_mask

            prefix_gold.extend(gold[pre_mask].tolist())
            prefix_pred.extend(pred[pre_mask].tolist())
            suffix_gold.extend(gold[suf_mask].tolist())
            suffix_pred.extend(pred[suf_mask].tolist())

            # Error localisation: argmax of abs_err over valid positions
            if valid_mask.any():
                err_with_nan = np.where(valid_mask, abs_err, -np.inf)
                argmax_pos = int(np.argmax(err_with_nan))
                if argmax_pos == ep:
                    loc_hits += 1
                loc_total += 1

        prefix_r2 = (r2_score(prefix_gold, prefix_pred)
                     if len(prefix_gold) > 1 else float("nan"))
        suffix_r2 = (r2_score(suffix_gold, suffix_pred)
                     if len(suffix_gold) > 1 else float("nan"))
        loc_acc = loc_hits / loc_total if loc_total > 0 else float("nan")

        per_layer_summary.append({
            "layer":     layer_idx,
            "prefix_r2": prefix_r2,
            "suffix_r2": suffix_r2,
            "loc_acc":   loc_acc,
            "n_strings": loc_total,
        })

    # ── 3. Report ──────────────────────────────────────────────────────────
    print("\n  Step 3/3: results")
    print(f"\n  {'Layer':<8} {'prefix R²':>11} {'suffix R²':>11} "
          f"{'loc-acc top-1':>15} {'n strings':>11}")
    print(f"  {'─' * 60}")
    for row in per_layer_summary:
        layer_label = "emb" if row["layer"] == 0 else f"L{row['layer']}"
        print(f"  {layer_label:<8} {row['prefix_r2']:>11.4f} {row['suffix_r2']:>11.4f} "
              f"{row['loc_acc']:>15.4f} {row['n_strings']:>11}")

    # Plot: prefix vs suffix R² per layer + localisation accuracy
    layers = [r["layer"] for r in per_layer_summary]
    labels = ["emb"] + [f"L{l}" for l in layers[1:]]
    prefix_r2s = [r["prefix_r2"] for r in per_layer_summary]
    suffix_r2s = [r["suffix_r2"] for r in per_layer_summary]
    loc_accs = [r["loc_acc"] for r in per_layer_summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(layers, prefix_r2s, marker="o", label="prefix (before error)",
             color="seagreen")
    ax1.plot(layers, suffix_r2s, marker="s", label="suffix (at/after error)",
             color="firebrick")
    ax1.set_xticks(layers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(r"Probe $R^2$ on corrupted test")
    ax1.set_title("Probe accuracy: prefix vs suffix")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(layers, loc_accs, marker="o", color="steelblue")
    ax2.set_xticks(layers)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Top-1 localisation accuracy")
    ax2.set_title("Error localisation: argmax(|pred − gold|) == error_position")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Q15: Probe trained on correct strings, evaluated on corrupted strings"
    )
    fig.tight_layout()
    path = out_dir / "q15_probe_divergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  plot saved → {path}")

    return per_layer_summary


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")

    train_loader = make_loader(args.train, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(args.test, batch_size=args.batch_size, shuffle=False)

    if args.correct_only and args.error_only:
        raise ValueError("--correct-only and --error-only are mutually exclusive")
    if args.correct_only:
        filter_mode, label = "correct", "correct_only"
    elif args.error_only:
        filter_mode, label = "error", "error_only"
    else:
        filter_mode, label = "all", "all"

    # ── Q13 ────────────────────────────────────────────────────────────────
    if args.q13:
        print("\nextracting CLS representations …")
        X_train, y_train, meta_train = extract_cls_representations(
            model, train_loader, device
        )
        X_test, y_test, meta_test = extract_cls_representations(
            model, test_loader, device
        )
        print(f"  train: X={X_train.shape}, y={y_train.shape}")
        print(f"  test:  X={X_test.shape},  y={y_test.shape}")

        if filter_mode != "all":
            train_mask = np.array([_record_passes_filter(m, filter_mode) for m in meta_train])
            test_mask = np.array([_record_passes_filter(m, filter_mode) for m in meta_test])
            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]
            print(f"  filtered to {filter_mode} only")
            print(f"    train: {X_train.shape},  test: {X_test.shape}")

        q13(X_train, y_train, X_test, y_test, out_dir, label)

    # ── Q14 ────────────────────────────────────────────────────────────────
    if args.q14:
        print(f"\nextracting per-layer per-token representations [{filter_mode}] …")
        reps_train, y_train_local, _ = extract_per_layer_representations(
            model, train_loader, device, filter_mode=filter_mode
        )
        reps_test, y_test_local, _ = extract_per_layer_representations(
            model, test_loader, device, filter_mode=filter_mode
        )
        print(f"  reps_train: {reps_train.shape}")
        print(f"  reps_test:  {reps_test.shape}")

        if args.min_gold_depth > 0:
            tr_mask = y_train_local >= args.min_gold_depth
            te_mask = y_test_local >= args.min_gold_depth
            reps_train = reps_train[:, tr_mask, :]
            y_train_local = y_train_local[tr_mask]
            reps_test = reps_test[:, te_mask, :]
            y_test_local = y_test_local[te_mask]
            print(f"  filtered to gold depth >= {args.min_gold_depth}")
            print(f"  reps_train: {reps_train.shape}")
            print(f"  reps_test:  {reps_test.shape}")

        q14(reps_train, y_train_local, reps_test, y_test_local, out_dir, label)

    # ── Q15 ────────────────────────────────────────────────────────────────
    if args.q15:
        q15(model, train_loader, test_loader, device, out_dir)


if __name__ == "__main__":
    main()
