"""
Training and Comparison Script: Run 1 (Zeus Baseline) vs Run 2 (MuSViT + Zeus).

Follows the Zeus repository workflow:
- Ingests datasets from Zeus pickled slices (ZeusDatasetSample).
- Run 1 (Zeus Baseline): CNN-BiLSTM Encoder + Zeus Bahdanau Attention Decoder (trained from scratch).
- Run 2 (MuSViT + Zeus): Pre-trained MuSViT Vision Transformer Feature Extractor + Zeus Decoder.
- Evaluation: Symbol Error Rate (SER) computed directly with zeus.evaluation.symbol_error_rate.
- Exports results to Markdown and CSV summary tables, merging concurrent SLURM runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from musvit_zeus_comparison.zeus.data.zeus_dataset import ZeusDatasetSample
    from musvit_zeus_comparison.zeus.evaluation.symbol_error_rate import symbol_error_rate
    from musvit_zeus_comparison.dataset import (
        StaveCollate,
        StaveOMRDataset,
        TokenVocabulary,
        load_zeus_pickles,
    )
    from musvit_zeus_comparison.models import CombinedOMRModel
except ImportError:
    from zeus.data.zeus_dataset import ZeusDatasetSample
    from zeus.evaluation.symbol_error_rate import symbol_error_rate
    from dataset import (
        StaveCollate,
        StaveOMRDataset,
        TokenVocabulary,
        load_zeus_pickles,
    )
    from models import CombinedOMRModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_ser(
    model: nn.Module,
    loader: DataLoader,
    vocab: TokenVocabulary,
    max_length: int,
    device: torch.device,
) -> float:
    """Computes Symbol Error Rate (SER) using greedy autoregressive generation and Zeus metric."""
    model.eval()
    gold_lmx_list: list[str] = []
    pred_lmx_list: list[str] = []

    with torch.no_grad():
        for inputs, _, targets in loader:
            inputs = inputs.to(device)
            generated = model.generate(inputs, max_length=max_length)

            for pred_seq, gold_seq in zip(generated.cpu().tolist(), targets.cpu().tolist()):
                clean_gold = [t for t in gold_seq if t not in (vocab.pad_idx, vocab.eos_idx, vocab.bos_idx)]
                clean_pred = []
                for t in pred_seq:
                    if t == vocab.eos_idx:
                        break
                    if t not in (vocab.pad_idx, vocab.bos_idx):
                        clean_pred.append(t)

                gold_lmx_list.append(" ".join(vocab.decode(clean_gold)))
                pred_lmx_list.append(" ".join(vocab.decode(clean_pred)))

    return symbol_error_rate(gold_lmx_list, pred_lmx_list)


def train_single_model(
    model_type: str,
    train_dataset: StaveOMRDataset,
    val_dataset: StaveOMRDataset,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
    test_dataset: StaveOMRDataset | None = None,
) -> dict[str, float | int | str]:
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
        timestep_width=getattr(args, "timestep_width", 16),
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        dropout=args.dropout,
    ).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = enc_params + dec_params
    print(f"Parameters: Encoder: {enc_params:,} | Decoder: {dec_params:,} | Total: {total_params:,}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint resuming
    start_epoch = 1
    best_val_loss = float("inf")
    best_ser = float("inf")

    resume_target = None
    if getattr(args, "resume", None):
        if str(args.resume).lower() == "auto":
            for cand in [output_dir / f"{model_type}_latest.pt", output_dir / f"{model_type}_best.pt"]:
                if cand.exists():
                    resume_target = cand
                    break
        else:
            cand = Path(args.resume)
            if cand.exists():
                resume_target = cand

    if resume_target is not None:
        print(f"Loading checkpoint from: {resume_target}")
        checkpoint = torch.load(resume_target, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            best_ser = checkpoint.get("ser", float("inf"))
        print(f"Resumed successfully. Training will continue from epoch {start_epoch:02d}/{args.epochs:02d}.")

    # Fast evaluation-only mode if requested
    if getattr(args, "eval_only", False):
        print(f"\nRunning standalone validation SER evaluation for {model_type}...")
        val_ser = evaluate_ser(model, val_loader, vocab, max_length=args.max_gen_length, device=device)
        print(f"Validation SER: {val_ser:.2f}%")

        test_ser = None
        if test_dataset is not None and len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
            )
            test_ser = evaluate_ser(model, test_loader, vocab, max_length=args.max_gen_length, device=device)
            print(f"Test SER: {test_ser:.2f}%")

        return {
            "model": "Zeus Baseline (Run 1)" if model_type == "zeus" else "MuSViT + Zeus (Run 2)",
            "model_type": model_type,
            "encoder": "CNN-BiLSTM" if model_type == "zeus" else "MuSViT + Adapter",
            "enc_params": enc_params,
            "dec_params": dec_params,
            "total_params": total_params,
            "train_loss": 0.0,
            "val_loss": round(best_val_loss, 4),
            "token_acc": 0.0,
            "ser": round(val_ser, 2),
            "test_ser": round(test_ser, 2) if test_ser is not None else None,
            "total_time_sec": 0.0,
            "avg_epoch_sec": 0.0,
            "history": [],
        }

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Optimizer matching TensorFlow Zeus specification
    opt_name = getattr(args, "optimizer", "adam").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate schedule
    if args.lr_decay == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    else:
        scheduler = None

    for _ in range(1, start_epoch):
        if scheduler:
            scheduler.step()

    start_time = time.time()
    history = []
    snapshots_dir = output_dir / "snapshots"
    if getattr(args, "save_snapshots", False):
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    if start_epoch > args.epochs:
        print(f"Model already reached epoch {start_epoch - 1} >= requested target epochs {args.epochs}. Skipping training loop.")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
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

            if scheduler:
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

            # Periodic Symbol Error Rate (SER) evaluation matching Zeus
            eval_each = getattr(args, "evaluation_each", 10)
            eval_from = getattr(args, "evaluation_from", 1)
            should_eval_ser = (eval_each > 0 and epoch >= eval_from and (epoch % eval_each == 0 or epoch == args.epochs))

            current_ser = None
            if should_eval_ser:
                current_ser = evaluate_ser(model, val_loader, vocab, max_length=args.max_gen_length, device=device)
                if current_ser < best_ser:
                    best_ser = current_ser
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": current_ser, "vocab": vocab.token2id},
                        output_dir / f"{model_type}_best.pt",
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not should_eval_ser:
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": best_ser, "vocab": vocab.token2id},
                        output_dir / f"{model_type}_best.pt",
                    )

            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": current_ser or best_ser, "vocab": vocab.token2id},
                output_dir / f"{model_type}_latest.pt",
            )

            if getattr(args, "save_snapshots", False) and should_eval_ser:
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": current_ser, "vocab": vocab.token2id},
                    snapshots_dir / f"{model_type}_epoch_{epoch:03d}.pt",
                )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "token_acc": token_acc,
                "ser": current_ser,
                "sec": epoch_sec,
            })

            ser_str = f" | Val SER: {current_ser:.2f}%" if current_ser is not None else ""
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Token Acc: {token_acc:.2f}%"
                f"{ser_str} | "
                f"Time: {epoch_sec:.1f}s"
            )

    total_training_time = time.time() - start_time

    # Final validation SER if not evaluated yet
    if best_ser == float("inf"):
        print(f"\nComputing Final Validation SER for {model_type}...")
        best_ser = evaluate_ser(model, val_loader, vocab, max_length=args.max_gen_length, device=device)
        print(f"Final Validation SER: {best_ser:.2f}%")
    else:
        print(f"\nBest Validation SER achieved during run: {best_ser:.2f}%")

    # Evaluate on Test Set if provided (matching Zeus test evaluation)
    test_ser = None
    if test_dataset is not None and len(test_dataset) > 0:
        print(f"\n=======================================================")
        print(f" Running Test Set Evaluation for {model_type}...")
        print(f"=======================================================")
        best_ckpt = output_dir / f"{model_type}_best.pt"
        if best_ckpt.exists():
            print(f"Loading best weights for test evaluation from: {best_ckpt}")
            ckpt_data = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt_data.get("model", ckpt_data))

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        test_ser = evaluate_ser(model, test_loader, vocab, max_length=args.max_gen_length, device=device)
        print(f"Final Test SER: {test_ser:.2f}%")

    final_train_loss = history[-1]["train_loss"] if history else 0.0
    final_token_acc = history[-1]["token_acc"] if history else 0.0
    elapsed_epochs = max(1, len(history))

    return {
        "model": "Zeus Baseline (Run 1)" if model_type == "zeus" else "MuSViT + Zeus (Run 2)",
        "model_type": model_type,
        "encoder": "CNN-BiLSTM" if model_type == "zeus" else "MuSViT + Adapter",
        "enc_params": enc_params,
        "dec_params": dec_params,
        "total_params": total_params,
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "token_acc": round(final_token_acc, 2),
        "ser": round(best_ser, 2),
        "test_ser": round(test_ser, 2) if test_ser is not None else None,
        "total_time_sec": round(total_training_time, 1),
        "avg_epoch_sec": round(total_training_time / elapsed_epochs, 2),
        "history": history,
    }


def export_results_table(results: list[dict], output_dir: str | Path):
    """Formats and writes a comparison table to Markdown, CSV, and stdout.
    Automatically merges with previously finished runs (e.g. when Run 1 and Run 2 are run in separate jobs).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save individual JSON results
    for r in results:
        m_type = r.get("model_type", "zeus" if "Zeus" in r.get("model", "") else "musvit")
        json_path = out_dir / f"{m_type}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            serializable = {k: v for k, v in r.items() if k != "history"}
            json.dump(serializable, f, indent=2)

    # 2. Check for previously saved counterpart run to merge
    all_runs: dict[str, dict] = {}
    for other_type in ["zeus", "musvit"]:
        cand_json = out_dir / f"{other_type}_results.json"
        if cand_json.exists():
            try:
                with open(cand_json, "r", encoding="utf-8") as f:
                    all_runs[other_type] = json.load(f)
            except Exception:
                pass

    for r in results:
        m_type = r.get("model_type", "zeus" if "Zeus" in r.get("model", "") else "musvit")
        all_runs[m_type] = r

    merged_results = list(all_runs.values())
    merged_results.sort(
        key=lambda x: 0 if "run 1" in str(x.get("model", "")).lower() or x.get("model_type") == "zeus" else 1
    )

    has_any_test = any(r.get("test_ser") is not None for r in merged_results)
    if has_any_test:
        headers = [
            "Model Run",
            "Encoder",
            "Enc Params",
            "Dec Params",
            "Total Params",
            "Val Loss (best)",
            "Token Acc (%)",
            "Val SER (%)",
            "Test SER (%)",
            "Train Time (s)",
        ]
    else:
        headers = [
            "Model Run",
            "Encoder",
            "Enc Params",
            "Dec Params",
            "Total Params",
            "Val Loss (best)",
            "Token Acc (%)",
            "SER (%)",
            "Train Time (s)",
        ]

    rows = []
    for r in merged_results:
        row = [
            r["model"],
            r["encoder"],
            f"{r['enc_params']:,}",
            f"{r['dec_params']:,}",
            f"{r['total_params']:,}",
            f"{r['val_loss']:.4f}",
            f"{r['token_acc']:.2f}%",
            f"{r['ser']:.2f}%",
        ]
        if has_any_test:
            row.append(f"{r['test_ser']:.2f}%" if r.get("test_ser") is not None else "N/A")
        row.append(f"{r['total_time_sec']:.1f}s")
        rows.append(row)

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
        if len(merged_results) >= 2:
            r1, r2 = merged_results[0], merged_results[1]
            val_ser_diff = r1["ser"] - r2["ser"]
            speed_ratio = r1["total_time_sec"] / max(1e-3, r2["total_time_sec"])
            if val_ser_diff > 0:
                f.write(f"- **MuSViT + Zeus outperforms Zeus Baseline by {val_ser_diff:.2f}% lower Validation SER!**\n")
            else:
                f.write(f"- **Zeus Baseline had {-val_ser_diff:.2f}% lower Validation SER than MuSViT + Zeus.**\n")

            if r1.get("test_ser") is not None and r2.get("test_ser") is not None:
                test_ser_diff = r1["test_ser"] - r2["test_ser"]
                if test_ser_diff > 0:
                    f.write(f"- **MuSViT + Zeus outperforms Zeus Baseline by {test_ser_diff:.2f}% lower Test SER!**\n")
                else:
                    f.write(f"- **Zeus Baseline had {-test_ser_diff:.2f}% lower Test SER than MuSViT + Zeus.**\n")

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


def resolve_split_paths(explicit_paths: list[str] | None, dataset_dir: str | Path | None, split: str) -> list[Path]:
    """Resolves dataset pickle files from explicit arguments or dataset directory."""
    if explicit_paths:
        return [Path(p) for p in explicit_paths]
    if dataset_dir:
        base = Path(dataset_dir)
        if base.is_dir():
            aliases = ["dev", "val", "validation"] if split in ["dev", "val", "validation"] else [split]
            found = []
            for s in aliases:
                found.extend(base.glob(f"samples.{s}.pickle"))
                found.extend(base.glob(f"*/samples.{s}.pickle"))
            found = sorted(list(set(found)))
            if found:
                return found
    return []


def main():
    parser = argparse.ArgumentParser(description="Train and compare Zeus Baseline vs MuSViT + Zeus using Zeus pickled datasets.")
    parser.add_argument(
        "--model",
        type=str,
        default="compare",
        choices=["zeus", "musvit", "compare"],
        help="Select 'zeus' (Run 1), 'musvit' (Run 2), or 'compare' (runs both consecutively).",
    )
    # Zeus CLI compatibility flags
    parser.add_argument(
        "--train",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to training dataset pickle(s), e.g. datasets/omniomr/samples.train.pickle (matches Zeus --train).",
    )
    parser.add_argument(
        "--dev",
        "--val",
        dest="dev",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to validation dataset pickle(s), e.g. datasets/omniomr/samples.dev.pickle (matches Zeus --dev).",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to test dataset pickle(s), e.g. datasets/omniomr/samples.test.pickle (matches Zeus --test).",
    )
    parser.add_argument("--dataset-dir", type=str, default="datasets", help="Base directory containing Zeus datasets (default: datasets).")
    parser.add_argument("--feature-cache-dir", type=str, default="feature_cache", help="Directory of pre-extracted MuSViT features.")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Directory to save logs, checkpoints and tables.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (Zeus default: 400-500).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (Zeus default: 32 or 64).")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3, help="Learning rate (Zeus default: 1e-3).")
    parser.add_argument("--lr-decay", type=str, default="cos", choices=["cos", "none"], help="Learning rate decay (Zeus default: cos).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer type ('adam' matching Zeus).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay (used only if optimizer is adamw).")
    parser.add_argument("--dim", type=int, default=256, help="Model hidden / embedding dimension.")
    parser.add_argument("--timestep-width", type=int, default=16, help="Timestep width for Zeus encoder (default: 16 matching solo26).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2 matching solo26).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--evaluation-from", type=int, default=1, help="Start evaluation with this epoch onward (matches Zeus).")
    parser.add_argument("--evaluation-each", "--eval-interval", dest="evaluation_each", type=int, default=10, help="Run validation evaluation every N epochs (matches Zeus).")
    parser.add_argument("--resume", type=str, default=None, help="Resume training. Pass 'auto' or path to .pt checkpoint.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save intermediate snapshot .pt checkpoints for evaluated epochs (matching Zeus).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only compute validation SER on loaded/existing checkpoint.")
    parser.add_argument("--max-gen-length", type=int, default=300, help="Max length for autoregressive evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not specified.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Active Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Resolve and load dataset pickles
    train_pickles = resolve_split_paths(args.train, args.dataset_dir, "train")
    dev_pickles = resolve_split_paths(args.dev, args.dataset_dir, "dev")
    test_pickles = resolve_split_paths(args.test, args.dataset_dir, "test")

    if not train_pickles:
        raise FileNotFoundError(
            f"No training pickles found. Specify --train <path.pickle> or provide a --dataset-dir containing samples.train.pickle.\n"
            f"To convert MusiCorpus datasets to Zeus pickles, run:\n"
            f"  zeus musicorpus --input <mc_dataset> --output datasets/<name> --take-staves\n"
            f"  zeus pickle datasets/<name>/samples.train.txt"
        )

    print("Loading training datasets...")
    train_samples = load_zeus_pickles(train_pickles)
    print("Loading validation datasets...")
    val_samples = load_zeus_pickles(dev_pickles) if dev_pickles else []
    print("Loading test datasets...")
    test_samples = load_zeus_pickles(test_pickles) if test_pickles else []

    # 2. Build shared vocabulary from training split (matching Zeus TokenMap)
    vocab = TokenVocabulary.build_from_samples(train_samples)
    vocab_path = Path(args.output_dir) / "vocab.json"
    if args.resume and vocab_path.exists():
        print(f"Loading existing vocabulary from {vocab_path} to preserve exact checkpoint token mapping...")
        vocab = TokenVocabulary.load(vocab_path)
    else:
        vocab.save(vocab_path)

    test_info = f" | {len(test_samples):,} Test" if test_samples else ""
    print(f"Data Split: {len(train_samples):,} Train | {len(val_samples):,} Validation{test_info}")
    print(f"Vocabulary: {len(vocab):,} unique LMX tokens.")

    models_to_run = ["zeus", "musvit"] if args.model == "compare" else [args.model]
    results = []

    for m in models_to_run:
        train_ds = StaveOMRDataset(train_samples, vocab=vocab, mode=m, feature_cache_dir=args.feature_cache_dir)
        val_ds = StaveOMRDataset(val_samples, vocab=vocab, mode=m, feature_cache_dir=args.feature_cache_dir)
        test_ds = StaveOMRDataset(test_samples, vocab=vocab, mode=m, feature_cache_dir=args.feature_cache_dir) if test_samples else None

        res = train_single_model(
            model_type=m,
            train_dataset=train_ds,
            val_dataset=val_ds,
            vocab=vocab,
            args=args,
            device=device,
            test_dataset=test_ds,
        )
        results.append(res)

    # 3. Export findings into table
    export_results_table(results, args.output_dir)


if __name__ == "__main__":
    main()
