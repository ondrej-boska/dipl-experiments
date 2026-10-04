"""
Training and Comparison Script: Run 1 (Zeus Baseline) vs Run 2 (MuSViT + Zeus).

Ensures 100% identical hyperparameters across runs (seed, splits, optimizer, scheduler, loss, decoder).
Evaluates Symbol Error Rate (SER / NED), loss, and speed, and automatically exports findings
into a formatted Markdown table and CSV file.
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

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


def compute_levenshtein_distance(seq1: list[int], seq2: list[int]) -> int:
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


def _index_feature_cache(cache_path: Path) -> dict[str, Path]:
    """
    Builds a fast in-memory index mapping sample identifiers to feature file paths in cache.
    Handles manifest lookups, exact filenames, stripped suffixes, and canonical MusiCorpus stave keys.
    """
    cache_index: dict[str, Path] = {}
    if not cache_path.exists():
        return cache_index

    # 1. Manifest index if manifest.json exists
    manifest_file = cache_path / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for entry in manifest_data:
                feat_p = Path(entry.get("feature_path", ""))
                if feat_p.exists():
                    if "image_path" in entry:
                        cache_index[str(Path(entry["image_path"]).resolve())] = feat_p
                    if "relative_path" in entry:
                        r_str = str(entry["relative_path"]).replace("\\", "_").replace("/", "_").replace(".jpg", "").replace(".png", "")
                        cache_index[r_str] = feat_p
        except Exception:
            pass

    # 2. Scan all .pt files in cache directory
    for pt_file in cache_path.glob("*.pt"):
        stem = pt_file.stem
        cache_index[pt_file.name] = pt_file
        cache_index[stem] = pt_file

        # Strip standard pooling and precision suffixes (e.g. _vertical_mean_float16)
        base_key = re.sub(r"_(vertical_mean|vertical_4slice|full)_(float16|float32)$", "", stem)
        cache_index[base_key] = pt_file

        # Index canonical MusiCorpus stave suffix (<piece_id>_Staves_<stave_id>_<stem>)
        # even if prefixed with dataset folders (e.g. 'OmniOMR_<piece>_Staves_<stave>_image')
        if "_Staves_" in base_key:
            parts = base_key.split("_Staves_")
            stave_after = parts[1]  # e.g. '10_image'
            piece_part = parts[0]
            cache_index[f"{piece_part}_Staves_{stave_after}"] = pt_file

            piece_tokens = piece_part.split("_")
            for i in range(1, len(piece_tokens)):
                stripped_piece = "_".join(piece_tokens[i:])
                cache_index[f"{stripped_piece}_Staves_{stave_after}"] = pt_file

    return cache_index


def find_feature_path(
    img_path: Path,
    dataset_path: Path,
    cache_path: Path,
    cache_index: dict[str, Path],
) -> Path:
    """
    Robustly resolves the pre-extracted feature file for an image.
    Supports single-dataset root, multi-dataset root (e.g. 'datasets/<subdataset>/...'),
    and arbitrary directory layouts.
    """
    # 1. Manifest lookup by absolute image path
    resolved_img_str = str(img_path.resolve())
    if resolved_img_str in cache_index:
        return cache_index[resolved_img_str]

    # 2. Canonical MusiCorpus stave key: <piece_id>_Staves_<stave_id>_<stem>
    if "Staves" in img_path.parts:
        parts = img_path.parts
        staves_idx = parts.index("Staves")
        if staves_idx > 0 and staves_idx + 1 < len(parts):
            piece_id = parts[staves_idx - 1]
            stave_id = parts[staves_idx + 1]
            stave_key = f"{piece_id}_Staves_{stave_id}_{img_path.stem}"

            if stave_key in cache_index:
                return cache_index[stave_key]

            cand = cache_path / f"{stave_key}_vertical_mean_float16.pt"
            if cand.is_file():
                return cand

    # 3. Exact relative path from dataset_dir
    rel_path = img_path.relative_to(dataset_path) if dataset_path in img_path.parents else Path(img_path.name)
    rel_str = str(rel_path).replace("\\", "_").replace("/", "_").replace(img_path.suffix, "")

    if rel_str in cache_index:
        return cache_index[rel_str]

    cand = cache_path / f"{rel_str}_vertical_mean_float16.pt"
    if cand.is_file():
        return cand

    # 4. If dataset_dir points to parent (e.g. 'datasets/'), strip top dataset folder
    if len(rel_path.parts) > 1:
        sub_rel = Path(*rel_path.parts[1:])
        sub_rel_str = str(sub_rel).replace("\\", "_").replace("/", "_").replace(img_path.suffix, "")
        if sub_rel_str in cache_index:
            return cache_index[sub_rel_str]
        cand = cache_path / f"{sub_rel_str}_vertical_mean_float16.pt"
        if cand.is_file():
            return cand

    # 5. Fallback path
    return cache_path / f"{rel_str}_vertical_mean_float16.pt"


def build_samples_and_vocab(dataset_dir: str | Path, feature_cache_dir: str | Path) -> tuple[list[dict], TokenVocabulary]:
    """Scans dataset, matches with feature cache, and builds shared vocabulary."""
    dataset_path = Path(dataset_dir)
    cache_path = Path(feature_cache_dir)

    staves = sorted(list(dataset_path.glob("*/Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(dataset_path.glob("*/*/Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(dataset_path.rglob("Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))

    cache_index = _index_feature_cache(cache_path)

    # Detect if LMX transcriptions are present in the dataset
    has_any_lmx = any((p.parent / "transcription.lmx").is_file() for p in staves[:min(len(staves), 500)])
    if not has_any_lmx:
        has_any_lmx = any((p.parent / "transcription.lmx").is_file() for p in staves)

    samples = []
    vocab = TokenVocabulary()
    matched_features = 0
    skipped_incompatible = 0

    print(f"Discovered {len(staves)} stave samples. Indexing features and building vocabulary...")
    for img_path in staves:
        lmx_path = img_path.parent / "transcription.lmx"
        musicxml_path = img_path.parent / "transcription.musicxml"

        # When LMX generation has been run, strictly require valid .lmx and skip unconvertible/errored staves
        if has_any_lmx:
            if not lmx_path.is_file() or lmx_path.stat().st_size == 0:
                skipped_incompatible += 1
                continue
            lmx_text = lmx_path.read_text(encoding="utf-8").strip()
            tokens = lmx_text.split()
            if not tokens:
                skipped_incompatible += 1
                continue
        else:
            # Fallback only for datasets where LMX generation was not run
            if lmx_path.is_file() and lmx_path.stat().st_size > 0:
                tokens = lmx_path.read_text(encoding="utf-8").strip().split()
            elif musicxml_path.is_file():
                tokens = extract_tokens_from_musicxml(musicxml_path)
            else:
                tokens = []
            if not tokens:
                skipped_incompatible += 1
                continue

        feat_path = find_feature_path(img_path, dataset_path, cache_path, cache_index)
        if feat_path.is_file():
            matched_features += 1

        entry = {
            "image_path": str(img_path.resolve()),
            "feature_path": str(feat_path.resolve()),
        }
        if lmx_path.is_file():
            entry["lmx_path"] = str(lmx_path.resolve())
        if musicxml_path.is_file():
            entry["musicxml_path"] = str(musicxml_path.resolve())

        for t in tokens:
            vocab.add_token(t)

        samples.append(entry)

    if skipped_incompatible > 0:
        print(f"LMX Consistency Filter: Skipped {skipped_incompatible} incompatible / errored samples without valid .lmx files.")
    print(f"Retained {len(samples)} valid samples for training and validation.")
    print(f"Feature Cache: matched {matched_features}/{len(samples)} samples to pre-extracted features in '{cache_path}'.")
    if matched_features < len(samples):
        print(f"  (Note: {len(samples) - matched_features} samples without features. Run 1 'zeus' will still work; Run 2 'musvit' requires extract_features.py).")
    print(f"Vocabulary initialized with {len(vocab)} unique tokens.")
    return samples, vocab


def evaluate_ser(
    model: nn.Module,
    val_loader: DataLoader,
    vocab: TokenVocabulary,
    max_length: int,
    device: torch.device,
) -> tuple[float, float, float, float, int, int]:
    """
    Computes Symbol Error Rate (SER) on the validation set using greedy autoregressive generation.
    Returns:
        ser: total edit distance / total reference tokens * 100
        avg_pred_len: average length of predicted token sequences (excluding BOS/EOS/PAD)
        avg_gold_len: average length of gold token sequences
        eos_rate: percentage of validation samples that emitted <eos> within max_length
        total_edit_distance: raw integer edit distance
        total_ref_tokens: raw integer reference token count
    """
    model.eval()
    total_edit_distance = 0
    total_ref_tokens = 0
    total_pred_tokens = 0
    eos_emitted_count = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, _, targets in val_loader:
            inputs = inputs.to(device)
            generated = model.generate(inputs, max_length=max_length)

            for pred_seq, gold_seq in zip(generated.cpu().tolist(), targets.cpu().tolist()):
                sample_count += 1
                clean_gold = [t for t in gold_seq if t not in (vocab.pad_idx, vocab.eos_idx, vocab.bos_idx)]
                clean_pred = []
                hit_eos = False
                for t in pred_seq:
                    if t == vocab.eos_idx:
                        hit_eos = True
                        break
                    if t not in (vocab.pad_idx, vocab.bos_idx):
                        clean_pred.append(t)

                if hit_eos:
                    eos_emitted_count += 1

                dist = compute_levenshtein_distance(clean_pred, clean_gold)
                total_edit_distance += dist
                total_ref_tokens += max(1, len(clean_gold))
                total_pred_tokens += len(clean_pred)

    ser = (total_edit_distance / max(1, total_ref_tokens)) * 100.0
    avg_pred_len = total_pred_tokens / max(1, sample_count)
    avg_gold_len = total_ref_tokens / max(1, sample_count)
    eos_rate = (eos_emitted_count / max(1, sample_count)) * 100.0
    return ser, avg_pred_len, avg_gold_len, eos_rate, total_edit_distance, total_ref_tokens


def train_single_model(
    model_type: str,
    train_dataset: StaveOMRDataset,
    val_dataset: StaveOMRDataset,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Trains either Run 1 ('zeus') or Run 2 ('musvit') and returns metrics summary."""
    print(f"\n=======================================================")
    print(f" Starting Training: {'Run 1: Zeus Baseline' if model_type == 'zeus' else 'Run 2: MuSViT + Zeus'}")
    print(f"=======================================================")

    # Verify MuSViT pre-extracted features exist before training
    if model_type == "musvit":
        missing_train = [s for s in train_dataset.samples if not Path(s.get("feature_path", "")).is_file()]
        missing_val = [s for s in val_dataset.samples if not Path(s.get("feature_path", "")).is_file()]
        total_missing = len(missing_train) + len(missing_val)
        if total_missing > 0:
            example_missing = (missing_train[0] if missing_train else missing_val[0]).get("feature_path", "N/A")
            raise FileNotFoundError(
                f"\n[Error] MuSViT training requires pre-extracted feature files in '{args.feature_cache_dir}'.\n"
                f"Missing feature files for {len(missing_train)}/{len(train_dataset)} training samples "
                f"and {len(missing_val)}/{len(val_dataset)} validation samples.\n"
                f"Example missing target: {example_missing}\n"
                f"Please run 'python -m musvit_zeus_comparison.extract_features' to extract features before training MuSViT."
            )

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

    # Checkpoint resuming
    start_epoch = 1
    best_val_loss = float("inf")
    best_ser = float("inf")

    resume_target = None
    if getattr(args, "resume", None):
        if str(args.resume).lower() == "auto":
            # Check latest first, then best
            for cand in [Path(args.output_dir) / f"{model_type}_latest.pt", Path(args.output_dir) / f"{model_type}_best.pt"]:
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
        ser, avg_pred, avg_gold, eos_pct, total_edit, total_ref = evaluate_ser(
            model, val_loader, vocab, max_length=args.max_gen_length, device=device
        )
        print(
            f"Validation SER: {ser:.2f}% "
            f"(Edit Dist: {total_edit:,} / {total_ref:,} tokens | "
            f"Avg Pred Len: {avg_pred:.1f} vs Gold: {avg_gold:.1f} | "
            f"EOS Rate: {eos_pct:.1f}%)"
        )
        return {
            "model": "Zeus Baseline (Run 1)" if model_type == "zeus" else "MuSViT + Zeus (Run 2)",
            "encoder": "CNN-BiLSTM" if model_type == "zeus" else "MuSViT + Adapter",
            "enc_params": enc_params,
            "dec_params": dec_params,
            "total_params": total_params,
            "train_loss": 0.0,
            "val_loss": round(best_val_loss, 4),
            "token_acc": 0.0,
            "ser": round(ser, 2),
            "total_time_sec": 0.0,
            "avg_epoch_sec": 0.0,
            "history": [],
        }

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Optimizer matching TensorFlow Zeus specification
    opt_name = getattr(args, "optimizer", "adam").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing learning rate schedule matching Zeus's CosineDecay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    for _ in range(1, start_epoch):
        scheduler.step()

    start_time = time.time()
    history = []
    output_dir = Path(args.output_dir)
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

            scheduler.step()
            train_loss /= len(train_dataset)

            # Validation phase (teacher-forcing token accuracy & loss)
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
            eval_interval = getattr(args, "eval_interval", 10)
            should_eval_ser = (eval_interval > 0 and (epoch % eval_interval == 0 or epoch == args.epochs))

            current_ser = None
            if should_eval_ser:
                ser, avg_pred, avg_gold, eos_pct, total_edit, total_ref = evaluate_ser(
                    model, val_loader, vocab, max_length=args.max_gen_length, device=device
                )
                current_ser = ser
                if ser < best_ser:
                    best_ser = ser
                    # Save best SER checkpoint
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": ser, "vocab": vocab.token2id},
                        output_dir / f"{model_type}_best.pt",
                    )

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not should_eval_ser:
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "ser": best_ser, "vocab": vocab.token2id},
                        output_dir / f"{model_type}_best.pt",
                    )

            # Always save latest checkpoint for resume safety
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

            if epoch % max(1, args.log_interval) == 0 or epoch == args.epochs:
                ser_str = f" | Val SER: {current_ser:.2f}% (EOS: {eos_pct:.1f}%)" if current_ser is not None else ""
                print(
                    f"Epoch {epoch:02d}/{args.epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Token Acc: {token_acc:.2f}%"
                    f"{ser_str} | "
                    f"Time: {epoch_sec:.1f}s"
                )

    total_training_time = time.time() - start_time

    # Final Symbol Error Rate (SER) computation if not computed in the final epoch
    if best_ser == float("inf"):
        print(f"\nComputing Final Symbol Error Rate (SER) for {model_type}...")
        ser, avg_pred, avg_gold, eos_pct, total_edit, total_ref = evaluate_ser(
            model, val_loader, vocab, max_length=args.max_gen_length, device=device
        )
        best_ser = ser
        print(
            f"Final Validation SER: {ser:.2f}% "
            f"(Total Edit Distance: {total_edit:,} / {total_ref:,} tokens | "
            f"Avg Pred Len: {avg_pred:.1f} vs Gold: {avg_gold:.1f} | "
            f"EOS Rate: {eos_pct:.1f}%)"
        )
    else:
        print(f"\nBest Validation SER achieved during run: {best_ser:.2f}%")

    final_train_loss = history[-1]["train_loss"] if history else 0.0
    final_token_acc = history[-1]["token_acc"] if history else 0.0
    elapsed_epochs = max(1, len(history))

    return {
        "model": "Zeus Baseline (Run 1)" if model_type == "zeus" else "MuSViT + Zeus (Run 2)",
        "encoder": "CNN-BiLSTM" if model_type == "zeus" else "MuSViT + Adapter",
        "enc_params": enc_params,
        "dec_params": dec_params,
        "total_params": total_params,
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "token_acc": round(final_token_acc, 2),
        "ser": round(best_ser, 2),
        "total_time_sec": round(total_training_time, 1),
        "avg_epoch_sec": round(total_training_time / elapsed_epochs, 2),
        "history": history,
    }


def export_results_table(results: list[dict], output_dir: str | Path):
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
    parser.add_argument("--dataset-dir", type=str, default="UFAL.OmniOMR", help="Dataset directory (default: UFAL.OmniOMR).")
    parser.add_argument("--feature-cache-dir", type=str, default="feature_cache", help="Directory of pre-extracted MuSViT features.")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Directory to save logs, checkpoints and tables.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (Zeus trains 400-500 from scratch).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (Zeus uses 32 or 64).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (Zeus default: 1e-3).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer type ('adam' matching Zeus, or 'adamw').")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay (used only if optimizer is adamw).")
    parser.add_argument("--dim", type=int, default=256, help="Model hidden / embedding dimension.")
    parser.add_argument("--timestep-width", type=int, default=16, help="Timestep width for Zeus encoder (default: 16 matching solo26).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2 matching solo26).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of samples for validation.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch loss print interval.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Interval (epochs) to evaluate validation SER (matches Zeus --evaluation-each 10).")
    parser.add_argument("--resume", type=str, default=None, help="Resume training. Pass 'auto' to auto-detect best/latest checkpoint, or path to .pt checkpoint.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save intermediate snapshot .pt checkpoints for evaluated epochs (matching Zeus).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only compute validation SER on loaded/existing checkpoint.")
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
    if args.resume and vocab_path.exists():
        print(f"Loading existing vocabulary from {vocab_path} to preserve exact checkpoint token mapping...")
        vocab = TokenVocabulary.load(vocab_path)
    else:
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
