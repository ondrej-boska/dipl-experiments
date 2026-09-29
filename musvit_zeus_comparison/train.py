"""
Training and Comparison Script: Run 1 (Zeus Baseline) vs Run 2 (MuSViT + Zeus).

Ensures 100% identical hyperparameters across runs (seed, splits, optimizer, scheduler, loss, decoder).
Evaluates Symbol Error Rate (SER / NED), loss, and speed, and automatically exports findings
into a formatted Markdown table and CSV file.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import (
    StaveCollate,
    StaveOMRDataset,
    TokenVocabulary,
    extract_tokens_from_musicxml,
)
from .models import CombinedOMRModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Computes standard edit distance between two integer token sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def build_samples_and_vocab(dataset_dir: str | Path, feature_cache_dir: str | Path) -> Tuple[List[dict], TokenVocabulary]:
    """Scans dataset, matches with feature cache, and builds shared vocabulary."""
    dataset_path = Path(dataset_dir)
    cache_path = Path(feature_cache_dir)

    staves = sorted(list(dataset_path.glob("*/Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(dataset_path.rglob("Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))

    samples = []
    vocab = TokenVocabulary()

    print(f"Discovered {len(staves)} stave samples. Building vocabulary...")
    for img_path in staves:
        rel_path = img_path.relative_to(dataset_path) if dataset_path in img_path.parents else img_path.name
        rel_str = str(rel_path).replace("\\", "_").replace("/", "_").replace(".jpg", "").replace(".png", "")

        # Look for matching feature file in cache (check common naming patterns)
        feat_path = cache_path / f"{rel_str}_vertical_mean_float16.pt"
        if not feat_path.exists():
            matches = list(cache_path.glob(f"{rel_str}*.pt"))
            feat_path = matches[0] if matches else feat_path

        entry = {
            "image_path": str(img_path.resolve()),
            "feature_path": str(feat_path.resolve()),
        }

        # Transcriptions
        musicxml_path = img_path.parent / "transcription.musicxml"
        if musicxml_path.exists():
            entry["musicxml_path"] = str(musicxml_path.resolve())
            tokens = extract_tokens_from_musicxml(musicxml_path)
            for t in tokens:
                vocab.add_token(t)

        lmx_path = img_path.parent / "transcription.lmx"
        if lmx_path.exists():
            entry["lmx_path"] = str(lmx_path.resolve())
            tokens = lmx_path.read_text(encoding="utf-8").strip().split()
            for t in tokens:
                vocab.add_token(t)

        samples.append(entry)

    print(f"Vocabulary initialized with {len(vocab)} unique tokens.")
    return samples, vocab


def train_single_model(
    model_type: str,
    train_dataset: StaveOMRDataset,
    val_dataset: StaveOMRDataset,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Union[float, int, str]]:
    """Trains either Run 1 ('zeus') or Run 2 ('musvit') and returns metrics summary."""
    print(f"\n=======================================================")
    print(f" Starting Training: {'Run 1: Zeus Baseline' if model_type == 'zeus' else 'Run 2: MuSViT + Zeus'}")
    print(f"=======================================================")

    collate_fn = StaveCollate(pad_idx=vocab.pad_idx, bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = CombinedOMRModel(
        encoder_type=model_type,
        vocab_size=len(vocab),
        dim=args.dim,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        dropout=args.dropout,
    ).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = enc_params + dec_params
    print(f"Parameters: Encoder: {enc_params:,} | Decoder: {dec_params:,} | Total: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val_loss = float("inf")
    start_time = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_tokens = 0

        for inputs, input_seqs, targets in train_loader:
            inputs = inputs.to(device)
            input_seqs = input_seqs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs, input_seqs)  # (B, L, V)

            loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_tokens += (targets != vocab.pad_idx).sum().item()

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, input_seqs, targets in val_loader:
                inputs = inputs.to(device)
                input_seqs = input_seqs.to(device)
                targets = targets.to(device)

                logits = model(inputs, input_seqs)
                loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
                val_loss += loss.item() * targets.size(0)

                preds = torch.argmax(logits, dim=-1)
                mask = (targets != vocab.pad_idx)
                val_correct += ((preds == targets) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_dataset)
        token_acc = (val_correct / max(1, val_total)) * 100.0
        epoch_sec = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "token_acc": token_acc,
            "sec": epoch_sec,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint
            ckpt_path = Path(args.output_dir) / f"{model_type}_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab.token2id}, ckpt_path)

        if epoch % max(1, args.log_interval) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Token Acc: {token_acc:.2f}% | "
                f"Time: {epoch_sec:.1f}s"
            )

    total_training_time = time.time() - start_time

    # Final Symbol Error Rate (SER / NED) evaluation on validation set using greedy generation
    print(f"\nComputing Final Symbol Error Rate (SER) for {model_type}...")
    model.eval()
    total_edit_distance = 0
    total_ref_tokens = 0

    with torch.no_grad():
        for inputs, _, targets in val_loader:
            inputs = inputs.to(device)
            # Greedy autoregressive generation
            generated = model.generate(inputs, max_length=args.max_gen_length)

            for pred_seq, gold_seq in zip(generated.cpu().tolist(), targets.cpu().tolist()):
                # Filter out BOS, EOS, PAD
                clean_gold = [t for t in gold_seq if t not in (vocab.pad_idx, vocab.eos_idx, vocab.bos_idx)]
                clean_pred = []
                for t in pred_seq:
                    if t == vocab.eos_idx:
                        break
                    if t not in (vocab.pad_idx, vocab.bos_idx):
                        clean_pred.append(t)

                dist = compute_levenshtein_distance(clean_pred, clean_gold)
                total_edit_distance += dist
                total_ref_tokens += max(1, len(clean_gold))

    ser = (total_edit_distance / max(1, total_ref_tokens)) * 100.0
    print(f"Final Validation SER: {ser:.2f}% (Total Edit Distance: {total_edit_distance} / {total_ref_tokens} tokens)")

    return {
        "model": "Zeus Baseline (Run 1)" if model_type == "zeus" else "MuSViT + Zeus (Run 2)",
        "encoder": "CNN-BiLSTM" if model_type == "zeus" else "MuSViT + Adapter",
        "enc_params": enc_params,
        "dec_params": dec_params,
        "total_params": total_params,
        "train_loss": round(history[-1]["train_loss"], 4),
        "val_loss": round(best_val_loss, 4),
        "token_acc": round(history[-1]["token_acc"], 2),
        "ser": round(ser, 2),
        "total_time_sec": round(total_training_time, 1),
        "avg_epoch_sec": round(total_training_time / max(1, args.epochs), 2),
        "history": history,
    }


def export_results_table(results: List[Dict], output_dir: str | Path):
    """Formats and writes a comparison table to Markdown, CSV, and stdout."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "Model Run",
        "Encoder",
        "Enc Params",
        "Dec Params",
        "Total Params",
        "Val Loss (best)",
        "Token Acc (%)",
        "SER / NED (%)",
        "Train Time (s)",
    ]

    rows = []
    for r in results:
        rows.append([
            r["model"],
            r["encoder"],
            f"{r['enc_params']:,}",
            f"{r['dec_params']:,}",
            f"{r['total_params']:,}",
            f"{r['val_loss']:.4f}",
            f"{r['token_acc']:.2f}%",
            f"{r['ser']:.2f}%",
            f"{r['total_time_sec']:.1f}s",
        ])

    # 1. Print formatted ASCII table to console
    col_widths = [max(len(str(val)) for val in [h] + [r[i] for r in rows]) + 2 for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    print("\n" + sep)
    print("|" + "|".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + "|")
    print(sep)
    for r in rows:
        print("|" + "|".join(r[i].center(col_widths[i]) for i in range(len(r))) + "|")
    print(sep + "\n")

    # 2. Write Markdown table
    md_path = out_dir / "results_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MuSViT vs Zeus Benchmark Comparison\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
        f.write("\n### Key Takeaways:\n")
        if len(results) >= 2:
            r1, r2 = results[0], results[1]
            ser_diff = r1["ser"] - r2["ser"]
            speed_ratio = r1["total_time_sec"] / max(1e-3, r2["total_time_sec"])
            if ser_diff > 0:
                f.write(f"- **MuSViT + Zeus outperforms Zeus Baseline by {ser_diff:.2f}% lower Symbol Error Rate!**\n")
            else:
                f.write(f"- **Zeus Baseline had {-ser_diff:.2f}% lower SER than MuSViT + Zeus.**\n")
            f.write(f"- Pre-extracted MuSViT training speed ratio: **{speed_ratio:.2f}x** relative to image CNN.\n")

    # 3. Write CSV
    csv_path = out_dir / "results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Results successfully saved:")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV:      {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare Zeus Baseline vs MuSViT + Zeus.")
    parser.add_argument(
        "--model",
        type=str,
        default="compare",
        choices=["zeus", "musvit", "compare"],
        help="Select 'zeus' (Run 1), 'musvit' (Run 2), or 'compare' (runs both consecutively).",
    )
    parser.add_argument("--dataset-dir", type=str, default="OmniOMR.Small", help="Dataset directory.")
    parser.add_argument("--feature-cache-dir", type=str, default="feature_cache", help="Directory of pre-extracted MuSViT features.")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Directory to save logs, checkpoints and tables.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--dim", type=int, default=256, help="Model hidden / embedding dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of samples for validation.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch print interval.")
    parser.add_argument("--max-gen-length", type=int, default=300, help="Max length for autoregressive evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not specified.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Active Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Discover samples and build shared vocabulary
    samples, vocab = build_samples_and_vocab(args.dataset_dir, args.feature_cache_dir)
    vocab_path = Path(args.output_dir) / "vocab.json"
    vocab.save(vocab_path)

    # 2. Partition identical train/val sample subsets
    total_samples = len(samples)
    val_size = max(1, int(total_samples * args.val_split))
    train_size = total_samples - val_size

    # Deterministic split via torch generator
    gen = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(range(total_samples), [train_size, val_size], generator=gen)

    train_samples = [samples[i] for i in train_indices.indices]
    val_samples = [samples[i] for i in val_indices.indices]
    print(f"Data Split: {len(train_samples)} Train | {len(val_samples)} Validation")

    models_to_run = ["zeus", "musvit"] if args.model == "compare" else [args.model]
    results = []

    for m in models_to_run:
        # Build dataset instances with identical sample distributions
        train_ds = StaveOMRDataset(train_samples, vocab=vocab, mode=m)
        val_ds = StaveOMRDataset(val_samples, vocab=vocab, mode=m)

        res = train_single_model(
            model_type=m,
            train_dataset=train_ds,
            val_dataset=val_ds,
            vocab=vocab,
            args=args,
            device=device,
        )
        results.append(res)

    # 3. Export findings into table
    export_results_table(results, args.output_dir)


if __name__ == "__main__":
    main()
