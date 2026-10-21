"""
Training and Comparison Script: Run 1 (Zeus Baseline) vs Run 2 (MuSViT + Zeus).

Follows the Zeus repository workflow:
- Ingests the predefined splits of the selected datasets from Zeus pickled slices (ZeusDatasetSample).
  Each split (train / dev / test) can combine any datasets found in the dataset directory.
- Run 1 (Zeus Baseline): CNN-BiLSTM Encoder + Zeus Bahdanau Attention Decoder (trained from scratch).
- Run 2 (MuSViT + Zeus): Pre-trained MuSViT Vision Transformer Feature Extractor + Zeus Decoder.
- Run 3 (end-to-end MuSViT + Zeus): the Run 2 model with the MuSViT backbone inside, reading stave images
  instead of cached features; the backbone is frozen, or fine-tuned (fully or with LoRA) from a given epoch on.
- Evaluation: Symbol Error Rate (SER) computed directly with zeus.evaluation.symbol_error_rate.
- Exports results to Markdown and CSV summary tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from musvit_zeus_comparison.zeus.evaluation.symbol_error_rate import symbol_error_rate, tokenize
    from musvit_zeus_comparison.augmentation import ZeusAugmentation
    from musvit_zeus_comparison.dataset import (
        LengthBucketBatchSampler,
        StaveCollate,
        StaveOMRDataset,
        TokenVocabulary,
        discover_datasets,
        feature_variant,
        load_split,
        missing_feature_files,
    )
    from musvit_zeus_comparison.models import CombinedOMRModel
    from musvit_zeus_comparison.predictions import write_predictions
except ImportError:
    from zeus.evaluation.symbol_error_rate import symbol_error_rate, tokenize
    from augmentation import ZeusAugmentation
    from dataset import (
        LengthBucketBatchSampler,
        StaveCollate,
        StaveOMRDataset,
        TokenVocabulary,
        discover_datasets,
        feature_variant,
        load_split,
        missing_feature_files,
    )
    from models import CombinedOMRModel
    from predictions import write_predictions


RUN_NAMES = {
    "zeus": "Zeus Baseline (Run 1)",
    "musvit": "MuSViT + Zeus (Run 2)",
    "musvit_e2e": "MuSViT E2E + Zeus (Run 3)",
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate_at(epoch: int, args: argparse.Namespace) -> float:
    """
    Learning rate of a (1-based) epoch. With 'cos' decay, it follows a cosine from the
    initial rate down towards 1% of it over all epochs. Being a pure function of the epoch,
    it stays correct when training resumes, even with a changed --epochs.
    """
    if args.lr_decay != "cos":
        return args.learning_rate
    min_lr = args.learning_rate * 0.01
    progress = (epoch - 1) / max(1, args.epochs)
    return min_lr + (args.learning_rate - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def finetunes_backbone(epoch: int, args: argparse.Namespace) -> bool:
    """Whether the MuSViT backbone of Run 3 is trained in a (1-based) epoch."""
    return args.finetune_from is not None and epoch >= args.finetune_from


def backbone_learning_rate_at(epoch: int, args: argparse.Namespace, progress: float = 1.0) -> float:
    """
    Learning rate of the MuSViT backbone after `progress` (0-1] of a (1-based) epoch: 0 while frozen, then
    --finetune-lr scaled by the same schedule as the rest of the model, so that it decays along with it.
    During the first --finetune-warmup-epochs of fine-tuning, it rises linearly from 0, step by step, so that
    the first updates of the pre-trained weights, made before Adam's moment estimates settle, stay small.
    """
    if not finetunes_backbone(epoch, args):
        return 0.0
    lr = args.finetune_lr * learning_rate_at(epoch, args) / args.learning_rate
    if args.finetune_warmup_epochs > 0:
        lr *= min(1.0, (epoch - args.finetune_from + progress) / args.finetune_warmup_epochs)
    return lr


def stave_width_arg(value: str) -> int | None:
    """--stave-width: a width in pixels, or 'aspect' (None) to keep each stave's aspect ratio."""
    if value.lower() == "aspect":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a width in pixels or 'aspect', got '{value}'") from None


def max_stave_width(args: argparse.Namespace) -> int:
    """Widest stave band of musvit_e2e, over which the position embeddings are stretched with 'interpolate'."""
    return args.stave_width or args.max_stave_width


def has_lora(state_dict: dict) -> bool:
    """Whether a model state dict holds LoRA adapters."""
    return any("lora_" in k for k in state_dict)


def check_finetune_method(model: CombinedOMRModel, state_dict: dict):
    """Fails clearly when a checkpoint to resume or evaluate was trained with another --finetune-method."""
    if model.backbone is None:
        return
    checkpoint_method = "lora" if has_lora(state_dict) else "full"
    if checkpoint_method != model.backbone.finetune_method:
        raise RuntimeError(
            f"The checkpoint was trained with --finetune-method {checkpoint_method}, so it must be resumed or "
            f"evaluated with it, not with '{model.backbone.finetune_method}'. To LoRA-tune a model trained without "
            f"LoRA, start a new run from it with --init-from."
        )


def model_config(model_type: str, vocab: TokenVocabulary, args: argparse.Namespace, musvit_dim: int = 768) -> dict:
    """
    What rebuilds the model for inference without the training data or arguments, saved in every checkpoint
    (see musibot_model.py): the CombinedOMRModel arguments under 'model' (the ViT configuration of musvit_e2e is
    added once the backbone is loaded) and the preprocessing of the input images under 'input'.
    """
    return {
        "model": {
            "encoder_type": model_type,
            "vocab_size": len(vocab),
            "dim": args.dim,
            "timestep_width": args.timestep_width,
            "input_height": args.image_height,
            "musvit_dim": musvit_dim,
            "bos_idx": vocab.bos_idx,
            "eos_idx": vocab.eos_idx,
            "pad_idx": vocab.pad_idx,
            "max_length": args.max_gen_length,
            "dropout": args.dropout,
            "musvit_model": args.musvit_model,
            "musvit_input": args.musvit_input,
            "stave_height": args.feature_stave_height,
            "feature_layout": args.feature_layout,
            "max_stave_width": max_stave_width(args),
            "finetune_method": args.finetune_method,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "input": {
            "image_height": args.image_height,
            "max_image_width": args.max_image_width,
            "stave_height": args.feature_stave_height,
            "stave_width": args.stave_width,
            "max_stave_width": args.max_stave_width,
        },
    }


def load_initial_weights(model: CombinedOMRModel, state_dict: dict):
    """
    Loads the weights of another training run (--init-from). A Run 2 checkpoint (cached features) has no
    backbone, which then keeps its pre-trained weights; all other weights must match. A Run 3 checkpoint without
    LoRA initializes a LoRA model too: its query/key/value weights are the base weights of the adapted projections,
    and the adapters start untrained.
    """
    if model.backbone is not None and model.backbone.finetune_method == "lora" and not has_lora(state_dict):
        state_dict = {re.sub(r"\.(query|key|value|[qkv]_proj)\.(weight|bias)$", r".\1.base_layer.\2", k): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("encoder.backbone.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"The --init-from checkpoint does not fit the '{model.encoder_type}' model. "
            f"Missing weights: {missing}. Unexpected weights: {result.unexpected_keys}."
        )
    if model.backbone is not None:
        pretrained = [k for k in result.missing_keys if "lora_" not in k]
        source = "pre-trained (not in the checkpoint)" if pretrained else "from the checkpoint"
        print(f"Initialized the model from the checkpoint; MuSViT backbone weights: {source}.")


def make_loader(dataset: StaveOMRDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    """
    Batches samples of similar transcription length (see LengthBucketBatchSampler).
    Unshuffled (evaluation) loaders iterate in length order, not dataset order.
    Shuffled (training) loaders augment the samples if the dataset has an augmentation.
    """
    vocab = dataset.vocab
    return DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(
            dataset.target_lengths(),
            args.batch_size,
            shuffle=shuffle,
            pool_batches=args.length_bucket_batches,
            sample_seeds=shuffle and dataset.augmentation is not None,
        ),
        collate_fn=StaveCollate(pad_idx=vocab.pad_idx, bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx),
        num_workers=args.num_workers,
        # Workers live across epochs instead of being re-forked (with the whole dataset) for every pass
        persistent_workers=args.num_workers > 0,
        # Page-locked batches can be copied to the GPU asynchronously
        pin_memory=torch.cuda.is_available(),
        # The worker seed comes from this generator instead of the global RNG. Persistent workers draw it only
        # when first started, which would otherwise shift the shuffling of a resumed run against an uninterrupted one.
        generator=torch.Generator().manual_seed(args.seed),
    )


def to_device(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moves a collated batch to the device. Input lengths stay on the CPU, where sequence packing needs them."""
    inputs, lengths, input_seqs, targets = batch
    return (
        inputs.to(device, non_blocking=True),
        lengths,
        input_seqs.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def evaluate_teacher_forced(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    vocab: TokenVocabulary,
    device: torch.device,
) -> tuple[float, float]:
    """Per-token loss and token accuracy (%) under teacher forcing."""
    model.eval()
    # Accumulated on the device, so that the CPU does not wait for the GPU after every batch
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in loader:
            inputs, lengths, input_seqs, targets = to_device(batch, device)
            logits = model(inputs, input_seqs, lengths)
            mask = targets != vocab.pad_idx
            n_tokens = mask.sum()
            loss_sum += criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) * n_tokens
            correct += ((logits.argmax(dim=-1) == targets) & mask).sum()
            total += n_tokens
    total_tokens = max(1, int(total))
    return loss_sum.item() / total_tokens, 100.0 * int(correct) / total_tokens


def predict_lmx(model: nn.Module, loader: DataLoader, vocab: TokenVocabulary, device: torch.device) -> list[str]:
    """Greedy autoregressive predictions as LMX strings, in the order of the loader's dataset."""
    batch_sampler = loader.batch_sampler
    assert isinstance(batch_sampler, LengthBucketBatchSampler) and not batch_sampler.shuffle, \
        "Predictions can only be put back in dataset order with a deterministic batch order."
    model.eval()
    pred_lmx_list: list[str] = [""] * len(loader.dataset)
    for batch, indices in zip(loader, batch_sampler, strict=True):
        inputs, lengths, _, _ = to_device(batch, device)
        generated = model.generate(inputs, lengths)
        for idx, pred_seq in zip(indices, generated.cpu().tolist(), strict=True):
            pred_lmx_list[idx] = vocab.decode_prediction(pred_seq)
    return pred_lmx_list


def evaluate_ser(model: nn.Module, loader: DataLoader, vocab: TokenVocabulary, device: torch.device) -> float:
    """Symbol Error Rate (SER) of greedy decoding against the gold LMX, using the Zeus metric."""
    return symbol_error_rate(loader.dataset.gold_lmx(), predict_lmx(model, loader, vocab, device))


def split_ser(dataset: StaveOMRDataset, preds: list[str]) -> tuple[float, dict[str, float]]:
    """
    SER of a split and of each of its source datasets. Each sample is compared only once: the SER of the split is
    the per-dataset ones weighted by their gold token counts, which is what the corpus-level SER sums to.
    """
    names, gold = dataset.dataset_names(), dataset.gold_lmx()
    per_dataset: dict[str, float] = {}
    gold_tokens: dict[str, int] = {}
    for name in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == name]
        per_dataset[name] = symbol_error_rate([gold[i] for i in idx], [preds[i] for i in idx])
        gold_tokens[name] = sum(len(tokenize(gold[i])) for i in idx)
    ser = sum(per_dataset[name] * gold_tokens[name] for name in per_dataset) / sum(gold_tokens.values())
    return ser, per_dataset


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab: TokenVocabulary,
    epoch: int,
    best: dict | None,
    history: list[dict],
    train_time_sec: float,
    config: dict,
):
    """Saves everything needed to evaluate the model or to resume its training exactly."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": vocab.to_dict(),
            "config": config,
            "epoch": epoch,
            "best": best,
            "history": history,
            "train_time_sec": train_time_sec,
            "rng": get_rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    # weights_only=False: own checkpoints also hold optimizer and RNG state (numpy arrays, tuples)
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_checkpoint(args: argparse.Namespace, model_type: str) -> Path | None:
    """The checkpoint to start from: an explicit --resume path, the best one for --eval-only, or the latest one."""
    output_dir = Path(args.output_dir)
    latest_path = output_dir / f"{model_type}_latest.pt"
    best_path = output_dir / f"{model_type}_best.pt"

    if args.resume and args.resume.lower() != "auto":
        path = Path(args.resume)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume from not found: '{path}'")
        return path

    if args.eval_only:
        if not best_path.is_file():
            raise FileNotFoundError(
                f"--eval-only needs a trained model, but '{best_path}' does not exist. "
                f"Pass --resume <checkpoint.pt> to evaluate a different checkpoint."
            )
        return best_path

    if args.resume:
        for candidate in (latest_path, best_path):
            if candidate.is_file():
                return candidate
        print(f"No checkpoint of '{model_type}' found in '{output_dir}'; training from scratch.")
    return None


def selection_metric(args: argparse.Namespace) -> str:
    """The checkpoint is selected by validation SER; by validation loss only when SER is never evaluated."""
    return "ser" if args.evaluation_each > 0 else "val_loss"


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe_encoder(model_type: str, args: argparse.Namespace) -> str:
    """Encoder column of the results table; for musvit_e2e with the backbone settings."""
    if model_type == "zeus":
        return "CNN-BiLSTM"
    if model_type == "musvit":
        return "MuSViT + Adapter"
    method = "LoRA " if args.finetune_method == "lora" else ""
    tuning = f"{method}fine-tuned from epoch {args.finetune_from}" if args.finetune_from is not None else "frozen"
    width = args.stave_width or f"aspect <= {args.max_stave_width}"
    return f"{args.musvit_model} ({args.musvit_input}, {args.feature_stave_height} x {width}, {tuning}) + Adapter"


def build_model(
    model_type: str,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
    musvit_dim: int,
    checkpoint: dict | None = None,
) -> tuple[CombinedOMRModel, dict]:
    """The model, and its configuration to save in checkpoints (see model_config)."""
    config = model_config(model_type, vocab, args, musvit_dim)
    # A resumed checkpoint holds all backbone weights (frozen ones too), so the pre-trained ones are not downloaded
    resumed_vit_config = ((checkpoint or {}).get("config") or {}).get("model", {}).get("vit_config")
    model = CombinedOMRModel(
        **config["model"],
        vit_config=resumed_vit_config,
        hf_token=args.hf_token,
        musvit_bf16=args.musvit_bf16,
        gradient_checkpointing=args.grad_checkpointing,
    ).to(device)

    enc_params, dec_params = n_params(model.encoder), n_params(model.decoder)
    encoder_str = f"{enc_params:,}"
    backbone = model.backbone
    if backbone is not None:
        # Builds the ViT without downloading the pre-trained weights, which the checkpoint replaces anyway
        config["model"]["vit_config"] = backbone.vit.config.to_dict()
        lora_str = ""
        if backbone.finetune_method == "lora":
            lora_str = f", of which LoRA adapters: {sum(p.numel() for p in backbone.adapter_parameters()):,}"
        encoder_str += f" (MuSViT backbone: {n_params(backbone):,}{lora_str})"
    print(f"Parameters: Encoder: {encoder_str} | Decoder: {dec_params:,} | Total: {enc_params + dec_params:,}")
    return model, config


def train(
    model: CombinedOMRModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
    config: dict,
    checkpoint: dict | None = None,
    init_checkpoint: dict | None = None,
) -> tuple[dict, list[dict], float]:
    """
    Trains the model and loads the weights of the selected epoch. `checkpoint` resumes a run exactly;
    `init_checkpoint` only provides the initial weights of a new run. Returns the metrics of the selected epoch
    (without those not evaluated during training), the history of all epochs, and the total training time.
    """
    backbone = model.backbone

    output_dir = Path(args.output_dir)
    latest_path = output_dir / f"{model_type}_latest.pt"
    best_path = output_dir / f"{model_type}_best.pt"
    snapshots_dir = output_dir / "snapshots"
    if args.save_snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Label smoothing applies to the training loss only; the validation loss stays plain cross-entropy,
    # comparable across runs
    train_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Optimizer matching TensorFlow Zeus specification; the fused CUDA kernel updates all parameters at once.
    # The MuSViT backbone gets a second parameter group with its own learning rate, from the start, so that the
    # optimizer state keeps one layout whether the backbone is frozen or not (frozen weights get no gradients,
    # which the optimizer skips).
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    param_groups = [{"params": [p for p in model.parameters() if id(p) not in backbone_ids]}]
    if backbone is not None:
        param_groups.append({"params": list(backbone.parameters())})
    fused = device.type == "cuda"
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=args.learning_rate, eps=1e-7, fused=fused)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay, fused=fused)

    select_by = selection_metric(args)

    start_epoch = 1
    best: dict | None = None
    history: list[dict] = []
    prior_train_time = 0.0

    if checkpoint is not None:
        check_finetune_method(model, checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        history = checkpoint["history"]
        prior_train_time = checkpoint["train_time_sec"]
        set_rng_state(checkpoint["rng"])
        print(f"Resumed from epoch {start_epoch - 1}.")
        if backbone is not None and history and history[-1].get("backbone_lr") and not finetunes_backbone(start_epoch, args):
            print(
                f"Warning: the backbone was fine-tuned before epoch {start_epoch}, "
                f"but stays frozen from now on, since --finetune-from is not given or is later."
            )
    elif init_checkpoint is not None:
        load_initial_weights(model, init_checkpoint["model"])

    if start_epoch > args.epochs:
        print(f"Model already reached epoch {start_epoch - 1} >= requested target epochs {args.epochs}. Skipping training loop.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        lr = learning_rate_at(epoch, args)
        optimizer.param_groups[0]["lr"] = lr
        backbone_lr = None
        if backbone is not None:
            finetune = finetunes_backbone(epoch, args)
            if finetune != backbone.trainable:
                print(f"Epoch {epoch}: {'unfreezing' if finetune else 'freezing'} the MuSViT backbone.")
            backbone.set_trainable(finetune)

        model.train()
        # Accumulated on the device, so that the CPU does not wait for the GPU after every batch
        train_loss_sum = torch.zeros((), device=device)
        train_tokens = torch.zeros((), dtype=torch.long, device=device)

        steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            inputs, lengths, input_seqs, targets = to_device(batch, device)
            if backbone is not None:
                # Set at every step, so that the warmup after unfreezing rises smoothly
                backbone_lr = backbone_learning_rate_at(epoch, args, (step + 1) / steps)
                optimizer.param_groups[1]["lr"] = backbone_lr

            optimizer.zero_grad()
            logits = model(inputs, input_seqs, lengths)  # (B, L, V)

            loss = train_criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_tokens = (targets != vocab.pad_idx).sum()
            train_loss_sum += loss.detach() * n_tokens
            train_tokens += n_tokens

        train_loss = train_loss_sum.item() / max(1, int(train_tokens))
        val_loss, token_acc = evaluate_teacher_forced(model, val_loader, criterion, vocab, device)

        # Periodic Symbol Error Rate (SER) evaluation matching Zeus
        should_eval_ser = (
            args.evaluation_each > 0
            and epoch >= args.evaluation_from
            and (epoch % args.evaluation_each == 0 or epoch == args.epochs)
        )
        current_ser = evaluate_ser(model, val_loader, vocab, device) if should_eval_ser else None

        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "token_acc": token_acc,
            "ser": current_ser,
            "sec": time.time() - epoch_start,
        }
        if backbone_lr is not None:
            record["backbone_lr"] = backbone_lr
        history.append(record)
        train_time = prior_train_time + time.time() - start_time

        improved = record[select_by] is not None and (best is None or record[select_by] < best[select_by])
        if improved:
            best = {k: record[k] for k in ("epoch", "train_loss", "val_loss", "token_acc", "ser")}
            save_checkpoint(best_path, model, optimizer, vocab, epoch, best, history, train_time, config)

        save_checkpoint(latest_path, model, optimizer, vocab, epoch, best, history, train_time, config)

        if args.save_snapshots and should_eval_ser:
            save_checkpoint(
                snapshots_dir / f"{model_type}_epoch_{epoch:03d}.pt",
                model, optimizer, vocab, epoch, best, history, train_time, config,
            )

        ser_str = f" | Val SER: {current_ser:.2f}%" if current_ser is not None else ""
        backbone_lr_str = f" | Backbone LR: {backbone_lr:.2e}" if backbone_lr else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"LR: {lr:.2e}{backbone_lr_str} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Token Acc: {token_acc:.2f}%"
            f"{ser_str} | "
            f"Time: {record['sec']:.1f}s"
            f"{' | * best' if improved else ''}"
        )

    total_training_time = prior_train_time + time.time() - start_time

    # All validation and test metrics are reported from one set of weights: the selected checkpoint
    if best is not None and best_path.is_file():
        print(f"\nLoading the selected checkpoint (epoch {best['epoch']}, by {select_by}) from: {best_path}")
        model.load_state_dict(load_checkpoint(best_path)["model"])
    else:
        print("\nNo checkpoint was selected during training; evaluating the final weights.")
        if history:
            best = {"epoch": history[-1]["epoch"], "train_loss": history[-1]["train_loss"]}
        else:
            best = {"epoch": start_epoch - 1}
    return best, history, total_training_time


def predictions_dir(model_type: str, args: argparse.Namespace) -> Path:
    """Folder of the predicted MusiCorpus datasets, one subfolder per source dataset."""
    return Path(args.predictions_dir) if args.predictions_dir else Path(args.output_dir) / f"{model_type}_predictions"


def save_split_predictions(
    dataset: StaveOMRDataset,
    preds: list[str],
    ser_per_dataset: dict[str, float],
    split: str,
    model_type: str,
    best: dict,
    args: argparse.Namespace,
):
    """
    Writes the predictions of a split as predicted MusiCorpus datasets, one per source dataset, for OMR-NED
    evaluation with omr-benchmark (see predictions.py). The splits of a dataset share its folder, since they hold
    different pages.
    """
    root = predictions_dir(model_type, args)
    print(f"\nWriting the {split} predictions as MusiCorpus datasets to '{root}'...")
    names, sample_names = dataset.dataset_names(), dataset.sample_names()
    for name in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == name]
        write_predictions(
            [sample_names[i] for i in idx],
            [preds[i] for i in idx],
            root / name,
            run_info={
                "model": RUN_NAMES[model_type],
                "model_type": model_type,
                "encoder": describe_encoder(model_type, args),
                "output_dir": args.output_dir,
                "epoch": best["epoch"],
                "dataset": name,
                # The MusiCorpus name of the split, which omr-benchmark --split takes as well
                "split": "validation" if split == "dev" else split,
                "ser": round(ser_per_dataset[name], 2),
            },
        )


def evaluate(
    model: CombinedOMRModel,
    model_type: str,
    best: dict,
    val_loader: DataLoader,
    test_dataset: StaveOMRDataset | None,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """
    Completes the validation metrics of the selected weights, whose known ones `best` holds, and computes the
    test SER, also per dataset (matching Zeus test evaluation). The predictions of the splits in
    --save-predictions are written as MusiCorpus datasets. Returns the rounded metrics.
    """
    if best.get("val_loss") is None:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        best["val_loss"], best["token_acc"] = evaluate_teacher_forced(model, val_loader, criterion, vocab, device)
    if best.get("ser") is None or "dev" in args.save_predictions:
        val_preds = predict_lmx(model, val_loader, vocab, device)
        val_ser, val_ser_per_dataset = split_ser(val_loader.dataset, val_preds)
        if best.get("ser") is None:
            best["ser"] = val_ser
        if "dev" in args.save_predictions:
            save_split_predictions(
                val_loader.dataset, val_preds, val_ser_per_dataset, "dev", model_type, best, args
            )
    print(
        f"Selected checkpoint: epoch {best['epoch']} | Val Loss: {best['val_loss']:.4f} | "
        f"Token Acc: {best['token_acc']:.2f}% | Val SER: {best['ser']:.2f}%"
    )

    test_ser = None
    test_ser_per_dataset: dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        print("\n=======================================================")
        print(" Running Test Set Evaluation...")
        print("=======================================================")
        preds = predict_lmx(model, make_loader(test_dataset, args, shuffle=False), vocab, device)
        test_ser, ser_per_dataset = split_ser(test_dataset, preds)
        print(f"Final Test SER: {test_ser:.2f}%")

        if len(ser_per_dataset) > 1:
            test_ser_per_dataset = ser_per_dataset
            for name, ser in test_ser_per_dataset.items():
                print(f"  - {name}: {ser:.2f}%")

        if "test" in args.save_predictions:
            save_split_predictions(test_dataset, preds, ser_per_dataset, "test", model_type, best, args)

    return {
        "best_epoch": best["epoch"],
        "train_loss": round(best["train_loss"], 4) if best.get("train_loss") is not None else None,
        "val_loss": round(best["val_loss"], 4),
        "token_acc": round(best["token_acc"], 2),
        "ser": round(best["ser"], 2),
        "test_ser": round(test_ser, 2) if test_ser is not None else None,
        "test_ser_per_dataset": {k: round(v, 2) for k, v in test_ser_per_dataset.items()},
    }


def export_results(result: dict, output_dir: Path):
    """Writes the results of a run as JSON and as a one-row table to Markdown (also printed) and CSV."""
    with open(output_dir / f"{result['model_type']}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    def fmt_pct(value) -> str:
        return f"{value:.2f}%" if value is not None else "N/A"

    per_dataset = result["test_ser_per_dataset"]
    headers = [
        "Model Run", "Encoder", "Enc Params", "Dec Params", "Total Params", "Best Epoch",
        "Val Loss", "Token Acc (%)", "Val SER (%)", "Test SER (%)",
        *(f"Test SER {name} (%)" for name in per_dataset),
        "Train Time (s)",
    ]
    row = [
        result["model"],
        result["encoder"],
        f"{result['enc_params']:,}",
        f"{result['dec_params']:,}",
        f"{result['total_params']:,}",
        str(result["best_epoch"]),
        f"{result['val_loss']:.4f}",
        fmt_pct(result["token_acc"]),
        fmt_pct(result["ser"]),
        fmt_pct(result["test_ser"]),
        *(fmt_pct(v) for v in per_dataset.values()),
        f"{result['total_time_sec']:.1f}s",
    ]

    markdown = "".join("| " + " | ".join(cells) + " |\n" for cells in (headers, ["---"] * len(headers), row))
    print("\n" + markdown)
    md_path = output_dir / "results_table.md"
    md_path.write_text(markdown, encoding="utf-8")

    csv_path = output_dir / "results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)

    print("Results successfully saved:")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV:      {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare Zeus Baseline vs MuSViT + Zeus using Zeus pickled datasets.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(RUN_NAMES),
        help="Select 'zeus' (Run 1), 'musvit' (Run 2, cached features), or 'musvit_e2e' (Run 3, MuSViT inside the "
             "model, optionally fine-tuned).",
    )
    # Dataset selection: every split uses the datasets' own predefined splits
    parser.add_argument("--dataset-dir", type=str, default="datasets", help="Directory containing the datasets as subfolders (default: datasets).")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets used for all splits unless overridden by --train/--dev/--test "
             "(default: all datasets in --dataset-dir).",
    )
    dataset_spec_help = (
        "Dataset(s) whose predefined '{}' split to use: names of subfolders of --dataset-dir, "
        "dataset folders, or .pickle files (default: {})."
    )
    parser.add_argument("--train", type=str, nargs="+", default=None, help=dataset_spec_help.format("train", "--datasets"))
    parser.add_argument(
        "--dev", "--val", dest="dev", type=str, nargs="+", default=None,
        help=dataset_spec_help.format("dev", "--datasets, or the --train datasets if --datasets is not given"),
    )
    parser.add_argument(
        "--test", type=str, nargs="*", default=None,
        help=dataset_spec_help.format("test", "--datasets, or the --train datasets if --datasets is not given")
             + " Pass --test without values to skip testing.",
    )
    parser.add_argument("--feature-cache-dir", type=str, default="feature_cache", help="Directory of pre-extracted MuSViT features.")
    parser.add_argument("--feature-stave-height", "--stave-height", dest="feature_stave_height", type=int, default=64, help="Stave height the MuSViT features were extracted with (extract_features.py --stave-height), or that musvit_e2e resizes staves to.")
    parser.add_argument("--feature-layout", type=str, default="columns", choices=["columns", "raster"], help="Sequence made of the MuSViT patch grid: one timestep per patch column with its rows concatenated ('columns'), or the row-major patch sequence of the MuSViT documentation ('raster').")
    parser.add_argument("--feature-precision", type=str, default="float16", choices=["float16", "float32"], help="Precision the MuSViT features were extracted with.")
    parser.add_argument("--preload-features", action="store_true", help="Read all MuSViT feature files into RAM once (~0.4 MB per stave at height 64) instead of from disk in every epoch; helps on slow (e.g. network) filesystems.")
    # Run 3 (musvit_e2e): MuSViT backbone inside the model
    parser.add_argument("--musvit-model", type=str, default="PRAIG/musvit", help="Hugging Face ID of the ViT backbone of musvit_e2e, e.g. 'PRAIG/musvit' or 'PRAIG/musvit-light' (loaded as a plain ViTModel).")
    parser.add_argument("--musvit-input", type=str, default="canvas", choices=["canvas", "interpolate"], help="How musvit_e2e feeds stave bands to the backbone: pasted on the white 1024x1024 canvas, exactly as extract_features.py ('canvas'), or alone with interpolated position embeddings, which the MuSViT model card recommends for fine-tuning and which processes 16x fewer patches at height 64 ('interpolate').")
    parser.add_argument("--stave-width", type=stave_width_arg, default=1024, help="Width musvit_e2e resizes staves to, a multiple of 16; must be the canvas size (1024) with --musvit-input canvas. 'aspect' keeps each stave's aspect ratio (with --musvit-input interpolate): the width follows the stave height, rounded to a multiple of 16 and capped at --max-stave-width; batches are padded with white and the padding is masked.")
    parser.add_argument("--max-stave-width", type=int, default=2048, help="Cap of the width of musvit_e2e staves with --stave-width aspect; longer staves are squashed. With --musvit-input interpolate, the pre-trained position embeddings are stretched over this width (with a fixed --stave-width, over that width).")
    parser.add_argument("--finetune-from", type=int, default=None, metavar="EPOCH", help="Fine-tune the musvit_e2e backbone from this (1-based) epoch on, frozen before; 1 fine-tunes from the start. Applies to resumed runs as well, e.g. resuming a frozen run at epoch 100 with --finetune-from 101. Default: frozen throughout.")
    parser.add_argument("--finetune-method", type=str, default="full", choices=["full", "lora"], help="How --finetune-from fine-tunes the musvit_e2e backbone: all its weights ('full'), or LoRA adapters on the attention query/key/value projections with the pre-trained weights frozen ('lora'), as the MuSViT repository does for staff-level OMR. A run must be resumed with the same method.")
    parser.add_argument("--finetune-lr", type=float, default=None, help="Learning rate of the backbone while fine-tuning, at the schedule's peak; it follows the --lr-decay schedule of --learning-rate, scaled. Default: 1e-5 with --finetune-method full, 3e-4 (the MuSViT repository's) with lora.")
    parser.add_argument("--lora-rank", type=int, default=8, help="Rank of the LoRA adapters (--finetune-method lora; MuSViT repository: 8).")
    parser.add_argument("--lora-alpha", type=float, default=16, help="LoRA alpha; the adapters are scaled by alpha / sqrt(rank) (rank-stabilized LoRA, as in the MuSViT repository: 16).")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="Dropout on the input of the LoRA adapters (MuSViT repository: 0.1).")
    parser.add_argument("--finetune-warmup-epochs", type=float, default=5, help="The backbone learning rate rises linearly from 0 over this many epochs (fractions allowed) after unfreezing, updated at every step; 0 disables the warmup.")
    parser.add_argument("--musvit-bf16", action="store_true", help="Run the musvit_e2e backbone in bfloat16 autocast (Ampere+ GPUs): much faster and less memory, slightly less precise.")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Recompute the backbone's activations in the backward pass instead of storing them, to fine-tune with less GPU memory at ~30%% more compute.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for the gated MuSViT models (default: the HF_TOKEN environment variable).")
    parser.add_argument("--init-from", type=str, default=None, help="Start a new training run (epoch 1, fresh optimizer) from the weights of this checkpoint. For musvit_e2e, a 'musvit' (Run 2) checkpoint with matching feature settings initializes the adapter and decoder, and the backbone keeps its pre-trained weights. Ignored when a checkpoint is resumed.")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Directory to save logs, checkpoints and tables.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (Zeus default: 400-500).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (Zeus default: 32 or 64).")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3, help="Learning rate (Zeus default: 1e-3).")
    parser.add_argument("--lr-decay", type=str, default="cos", choices=["cos", "none"], help="Learning rate decay (Zeus default: cos).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer type ('adam' matching Zeus).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay (used only if optimizer is adamw).")
    parser.add_argument("--dim", type=int, default=256, help="Model hidden / embedding dimension.")
    parser.add_argument("--timestep-width", type=int, default=16, help="Timestep width for Zeus encoder (default: 16 matching solo26).")
    parser.add_argument("--image-height", type=int, default=96, help="Height stave images are scaled to for the Zeus encoder.")
    parser.add_argument("--max-image-width", type=int, default=1536, help="Maximum width of stave images after scaling.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2 matching solo26).")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing of the training cross-entropy loss, e.g. 0.1; the validation loss is computed without it (default: 0, none).")
    parser.add_argument("--augment",type=str, default="", help="Training image augmentation, as Zeus --augment, e.g. 'h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2' (see augmentation.py); pixel amounts refer to the model's input image. For 'zeus' and 'musvit_e2e'. Default: none.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--length-bucket-batches", type=int, default=1, help="Training batches are formed from pools of this many batches sorted by transcription length, so that a batch holds similar lengths and the decoder wastes few steps on padding; 1 gives plain random batches.")
    parser.add_argument("--evaluation-from", type=int, default=1, help="Start evaluation with this epoch onward (matches Zeus).")
    parser.add_argument("--evaluation-each", "--eval-interval", dest="evaluation_each", type=int, default=10, help="Run validation SER evaluation every N epochs (matches Zeus). The best checkpoint is selected by it; 0 selects by validation loss instead.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training. Pass 'auto' or path to .pt checkpoint.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save intermediate snapshot .pt checkpoints for evaluated epochs (matching Zeus).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the best checkpoint (or the one given by --resume).")
    parser.add_argument("--save-predictions", type=str, nargs="*", default=["test"], metavar="SPLIT", help="Splits ('dev' and/or 'test') whose predictions of the selected checkpoint are written as predicted MusiCorpus datasets (LMX and the MusicXML decoded from it), for OMR-NED evaluation with omr-benchmark. Pass --save-predictions without values to write none (default: test).")
    parser.add_argument("--predictions-dir", type=str, default=None, help="Folder of the predicted MusiCorpus datasets, one subfolder per dataset (default: <output-dir>/<model>_predictions).")
    parser.add_argument("--max-gen-length", type=int, default=None, help="Max length for autoregressive decoding (default: 1.2x the longest training sequence).")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not specified.")
    parser.add_argument("--no-tf32", action="store_true", help="Compute matrix multiplications in full FP32 instead of TF32 (TF32 runs on the tensor cores of Ampere and newer GPUs).")

    args = parser.parse_args()
    if args.finetune_from is not None and args.finetune_from < 1:
        parser.error("--finetune-from must be an epoch >= 1.")
    if args.finetune_warmup_epochs < 0:
        parser.error("--finetune-warmup-epochs must be >= 0.")
    if args.finetune_lr is None:
        args.finetune_lr = 3e-4 if args.finetune_method == "lora" else 1e-5
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1.")
    split_names = {"dev": "dev", "val": "dev", "validation": "dev", "test": "test"}
    unknown_splits = [s for s in args.save_predictions if s not in split_names]
    if unknown_splits:
        parser.error(f"--save-predictions takes 'dev' and/or 'test', got: {', '.join(unknown_splits)}.")
    args.save_predictions = {split_names[s] for s in args.save_predictions}
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1).")
    if args.stave_width is None and args.model == "musvit_e2e" and args.musvit_input != "interpolate":
        parser.error("--stave-width aspect needs --musvit-input interpolate; the 'canvas' input is always 1024 px wide.")
    for name in ("stave_width", "max_stave_width"):
        width = getattr(args, name)
        if width is not None and (width <= 0 or width % 16 != 0):
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 16 (the patch size), got {width}.")
    if args.augment and args.model == "musvit":
        parser.error("--augment cannot augment the pre-extracted features of 'musvit'; use 'zeus' or 'musvit_e2e'.")
    try:
        ZeusAugmentation(args.augment)
    except ValueError as e:
        parser.error(f"--augment: {e}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Active Device: {device}")

    # TF32 tensor cores for matrix multiplications (decoder LSTM cell, attention, projections); PyTorch already
    # uses TF32 for convolutions and cuDNN LSTMs by default. No effect on the CPU or pre-Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = args.model

    # 1. Select datasets for each split and load their predefined splits.
    # Without --datasets, dev and test follow the training datasets rather than everything in --dataset-dir.
    train_specs = args.train or args.datasets or discover_datasets(args.dataset_dir)
    default_eval_specs = args.datasets or train_specs
    dev_specs = args.dev or default_eval_specs
    test_specs = args.test if args.test is not None else default_eval_specs

    if not train_specs:
        raise FileNotFoundError(
            f"No datasets selected and none found in '{args.dataset_dir}'.\n"
            f"To convert MusiCorpus datasets to Zeus pickles, run:\n"
            f"  python -m musvit_zeus_comparison.zeus musicorpus --input <mc_dataset> --output {args.dataset_dir}/<name> --take-staves\n"
            f"  python -m musvit_zeus_comparison.zeus pickle {args.dataset_dir}/<name>/samples.*.txt"
        )

    # The cached-feature MuSViT run alone reads pre-extracted features and never needs the images
    keep_images = model_type != "musvit"
    print("Loading training datasets...")
    train_samples = load_split(train_specs, args.dataset_dir, "train", keep_images)
    print("Loading validation datasets...")
    val_samples = load_split(dev_specs, args.dataset_dir, "dev", keep_images)
    print("Loading test datasets...")
    test_samples = load_split(test_specs, args.dataset_dir, "test", keep_images) if test_specs else {}

    selection = {"train": list(train_samples), "dev": list(val_samples), "test": list(test_samples)}
    for split, names in selection.items():
        print(f"  {split}: {', '.join(names) or '-'}")

    all_train = [s for samples in train_samples.values() for s in samples]
    n_val = sum(map(len, val_samples.values()))
    n_test = sum(map(len, test_samples.values()))
    if n_val == 0:
        raise ValueError("The validation split is empty; it is needed for checkpoint selection.")
    print(f"Data Split: {len(all_train):,} Train | {n_val:,} Validation | {n_test:,} Test")

    # Decoding must be able to produce the longest transcriptions, or SER is inflated by truncation
    max_train_len = max(len(s.lmx.split()) for s in all_train)
    if args.max_gen_length is None:
        args.max_gen_length = int(1.2 * max_train_len) + 1
    print(f"Max generation length: {args.max_gen_length} (longest training sequence: {max_train_len} tokens)")
    for split, samples in (("dev", val_samples), ("test", test_samples)):
        too_long = sum(len(s.lmx.split()) > args.max_gen_length for ss in samples.values() for s in ss)
        if too_long:
            print(f"Warning: {too_long:,} {split} sample(s) are longer than the max generation length and will be truncated.")

    variant = feature_variant(args.feature_stave_height, args.feature_precision)
    if model_type == "musvit":
        # Fail now rather than hours into training
        missing = [
            path
            for samples in (train_samples, val_samples, test_samples)
            for path in missing_feature_files(samples, args.feature_cache_dir, variant)
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing):,} MuSViT feature file(s) are missing, e.g. '{missing[0]}'. Run extract_features.py "
                f"with --stave-height {args.feature_stave_height} --precision {args.feature_precision} first."
            )

    checkpoint_path = resolve_checkpoint(args, model_type)
    checkpoint = None
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
    init_checkpoint = None
    if args.init_from and not args.eval_only:
        if checkpoint is not None:
            print(f"Resuming '{checkpoint_path}', so --init-from '{args.init_from}' is ignored.")
        else:
            print(f"Initializing weights from: {args.init_from}")
            init_checkpoint = load_checkpoint(Path(args.init_from))

    # 2. Vocabulary from the training split (matching Zeus TokenMap), or the checkpoint's exact mapping
    vocab_source = checkpoint or init_checkpoint
    if vocab_source is not None:
        vocab = TokenVocabulary.from_dict(vocab_source["vocab"])
        unknown = {t for s in all_train for t in s.lmx.split()} - vocab.token2id.keys()
        if unknown:
            print(f"Warning: {len(unknown):,} training token type(s) are not in the checkpoint's vocabulary and map to <unk>.")
    else:
        vocab = TokenVocabulary.build_from_samples(all_train)
    vocab.save(output_dir / f"{model_type}_vocab.json")
    print(f"Vocabulary: {len(vocab):,} unique LMX tokens.")

    dataset_kwargs = dict(
        vocab=vocab,
        mode=model_type,
        feature_cache_dir=args.feature_cache_dir,
        feature_variant=variant,
        feature_layout=args.feature_layout,
        preload_features=args.preload_features,
        image_height=args.image_height,
        max_image_width=args.max_image_width,
        stave_height=args.feature_stave_height,
        stave_width=args.stave_width,
        max_stave_width=args.max_stave_width,
    )
    train_ds = StaveOMRDataset(train_samples, **dataset_kwargs, augment=args.augment)
    val_ds = StaveOMRDataset(val_samples, **dataset_kwargs)
    test_ds = StaveOMRDataset(test_samples, **dataset_kwargs) if test_samples else None

    musvit_dim = train_ds[0][0].shape[-1] if model_type == "musvit" else 768

    # 3. Train (or load the checkpoint to evaluate) and evaluate the selected weights
    print("\n=======================================================")
    print(f" Starting Training: {RUN_NAMES[model_type]}")
    print("=======================================================")

    set_seed(args.seed)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    model, config = build_model(model_type, vocab, args, device, musvit_dim, checkpoint)

    if args.eval_only:
        check_finetune_method(model, checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        print(f"\nEvaluating the loaded checkpoint (epoch {checkpoint['epoch']})...")
        best, history, train_time = {"epoch": checkpoint["epoch"]}, checkpoint["history"], checkpoint["train_time_sec"]
    else:
        best, history, train_time = train(
            model, model_type, train_loader, val_loader, vocab, args, device, config, checkpoint, init_checkpoint
        )
    metrics = evaluate(model, model_type, best, val_loader, test_ds, vocab, args, device)

    enc_params, dec_params = n_params(model.encoder), n_params(model.decoder)
    export_results(
        {
            "model": RUN_NAMES[model_type],
            "encoder": describe_encoder(model_type, args),
            "model_type": model_type,
            "enc_params": enc_params,
            "dec_params": dec_params,
            "total_params": enc_params + dec_params,
            "select_by": selection_metric(args),
            **metrics,
            "epochs_trained": len(history),
            "total_time_sec": round(train_time, 1),
            "avg_epoch_sec": round(train_time / max(1, len(history)), 2),
            "datasets": selection,
            "augment": args.augment,
        },
        output_dir,
    )


if __name__ == "__main__":
    main()
