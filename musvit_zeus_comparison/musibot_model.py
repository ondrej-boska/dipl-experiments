"""
Musibot wrapper of the trained staff-level models: the Zeus baseline ('zeus', Run 1) and end-to-end
MuSViT + Zeus ('musvit_e2e', Run 3). The cached-feature model ('musvit', Run 2) cannot read images and is not wrapped.

Two commands:
- export: turns a training checkpoint into a self-contained model snapshot folder:
      <snapshot>/
        model.json   # Musibot name and version, the CombinedOMRModel arguments, the input preprocessing
        vocab.json   # the token vocabulary
        weights.pt   # the model's state dict, the (fine-tuned) MuSViT backbone included
  Nothing is downloaded when a snapshot is served: the MuSViT backbone is built from the ViT configuration in
  model.json, and all its weights come from weights.pt.
- serve: runs a snapshot as a Musibot Model, i.e. the subprocess a Musibot Worker Head starts and talks to over
  the worker IPC (see musibot/docs/worker-ipc.md). It reads one staff image and writes its transcription beside it:
      Staves/{s}/image.jpg -> Staves/{s}/transcription.musicxml, Staves/{s}/transcription.lmx

    python -m musvit_zeus_comparison.musibot_model export experiment_results_e2e/musvit_e2e_best.pt snapshots/musvit-e2e.model
    musibot-worker-head --model-command "python -m musvit_zeus_comparison.musibot_model serve --model-snapshot snapshots/musvit-e2e.model"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

import torch
import torch.nn.functional as F
from lmx.musicxml.io.write_musicxml_tree_to_file import write_musicxml_tree_to_file
from lmx.musicxml.part_to_score import part_to_score
from lmx.tokenization.Decoder import Decoder

try:
    from musvit_zeus_comparison.dataset import TokenVocabulary, decode_stave_band, decode_zeus_image
    from musvit_zeus_comparison.models import CombinedOMRModel
except ImportError:
    from dataset import TokenVocabulary, decode_stave_band, decode_zeus_image
    from models import CombinedOMRModel


IPC_VERSION = 1

SIGNATURE = {
    "input": ["Staves/{s}/image.jpg"],
    "output": ["Staves/{s}/transcription.musicxml", "Staves/{s}/transcription.lmx"],
}

DEFAULT_NAMES = {"zeus": "zeus-baseline", "musvit_e2e": "musvit-e2e"}

# train.py defaults, for checkpoints saved before the model configuration was stored in them
LEGACY_DEFAULTS = {
    "timestep_width": 16,
    "image_height": 96,
    "max_image_width": 1536,
    "musvit_model": "PRAIG/musvit",
    "musvit_input": "canvas",
    "stave_height": 64,
    "stave_width": 1024,
    "max_stave_width": 2048,
    "feature_layout": "columns",
}


# ==============================================================================
# Export: training checkpoint -> model snapshot
# ==============================================================================

def legacy_config(state_dict: dict, vocab: TokenVocabulary, args: argparse.Namespace) -> dict:
    """
    The model configuration of a checkpoint saved before train.py stored it, from the export arguments (train.py
    defaults where not given). The encoder type and dimension are read from the weights; everything that changes
    the weights' shapes is checked when the weights are loaded, but the preprocessing and the generation length
    are not, so they have to be given as the run was trained.
    """
    if any(k.startswith("encoder.backbone.") for k in state_dict):
        encoder_type = "musvit_e2e"
    elif any(k.startswith("encoder.conv.") for k in state_dict):
        encoder_type = "zeus"
    else:
        raise ValueError("This is a cached-feature MuSViT checkpoint ('musvit'), which cannot read images; "
                         "only 'zeus' and 'musvit_e2e' checkpoints can be exported.")
    if args.max_gen_length is None:
        raise ValueError("This checkpoint predates stored model configurations, so --max-gen-length is needed: "
                         "the 'Max generation length' that train.py printed for the run.")

    def setting(name: str):
        value = getattr(args, name)
        return LEGACY_DEFAULTS[name] if value is None else value

    stave_width = setting("stave_width")
    stave_width = None if stave_width == "aspect" else int(stave_width)
    max_stave_width = setting("max_stave_width")
    config = {
        "model": {
            "encoder_type": encoder_type,
            "vocab_size": len(vocab),
            "dim": state_dict["decoder.embedding.weight"].shape[1],
            "timestep_width": setting("timestep_width"),
            "input_height": setting("image_height"),
            "bos_idx": vocab.bos_idx,
            "eos_idx": vocab.eos_idx,
            "pad_idx": vocab.pad_idx,
            "max_length": args.max_gen_length,
            "musvit_model": setting("musvit_model"),
            "musvit_input": setting("musvit_input"),
            "stave_height": setting("stave_height"),
            "feature_layout": setting("feature_layout"),
            "max_stave_width": stave_width or max_stave_width,
        },
        "input": {
            "image_height": setting("image_height"),
            "max_image_width": setting("max_image_width"),
            "stave_height": setting("stave_height"),
            "stave_width": stave_width,
            "max_stave_width": max_stave_width,
        },
    }
    if encoder_type == "musvit_e2e":
        # Only the configuration is downloaded; all weights are in the checkpoint
        from transformers import ViTConfig
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        vit_config = ViTConfig.from_pretrained(config["model"]["musvit_model"], token=hf_token)
        config["model"]["vit_config"] = vit_config.to_dict()
    return config


def export_snapshot(args: argparse.Namespace):
    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"'{output}' exists and is not empty; snapshots are not overwritten.")

    # weights_only=False: own checkpoints also hold optimizer and RNG state (numpy arrays, tuples)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    vocab = TokenVocabulary.from_dict(checkpoint["vocab"])

    config = checkpoint.get("config")
    legacy_flags = [name for name in (*LEGACY_DEFAULTS, "max_gen_length") if getattr(args, name) is not None]
    if config is None:
        config = legacy_config(state_dict, vocab, args)
    elif legacy_flags:
        print(f"The checkpoint stores its configuration, so these arguments are ignored: {', '.join(legacy_flags)}")

    encoder_type = config["model"]["encoder_type"]
    if encoder_type not in DEFAULT_NAMES:
        raise ValueError(f"Only 'zeus' and 'musvit_e2e' checkpoints can be exported, this one is '{encoder_type}'.")

    # Building the model and loading the weights strictly checks the configuration against the weights
    model = CombinedOMRModel(**config["model"])
    model.load_state_dict(state_dict)

    epoch = checkpoint.get("epoch")
    trained = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
    info = {
        "name": args.name or DEFAULT_NAMES[encoder_type],
        "version": args.version or f"{trained:%Y-%m-%d-%H%M%S}" + (f"-e{epoch}" if epoch is not None else ""),
        "config": config,
        "source": {"checkpoint": str(checkpoint_path), "epoch": epoch, "best": checkpoint.get("best")},
    }

    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "weights.pt")
    vocab.save(output / "vocab.json")
    (output / "model.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    # The snapshot has to load the way serve loads it
    Transcriber(output, torch.device("cpu"))
    print(f"Exported '{checkpoint_path}' (epoch {epoch}) to '{output}' as {info['name']} {info['version']}.")


# ==============================================================================
# Inference on a snapshot
# ==============================================================================

class Transcriber:
    """A model snapshot loaded for inference: stave images in, LMX and MusicXML out."""

    def __init__(self, snapshot: Path, device: torch.device, bf16: bool = False):
        info = json.loads((snapshot / "model.json").read_text(encoding="utf-8"))
        self.name: str = info["name"]
        self.version: str = info["version"]
        self.input: dict = info["config"]["input"]
        self.encoder_type: str = info["config"]["model"]["encoder_type"]
        self.vocab = TokenVocabulary.load(snapshot / "vocab.json")
        self.device = device

        model = CombinedOMRModel(**info["config"]["model"], musvit_bf16=bf16)
        model.load_state_dict(torch.load(snapshot / "weights.pt", map_location="cpu", weights_only=True))
        self.model = model.to(device).eval()

    def prepare(self, image_bytes: bytes) -> torch.Tensor:
        """An encoded stave image -> the model's input, preprocessed exactly as in training."""
        if self.encoder_type == "zeus":
            return decode_zeus_image(image_bytes, self.input["image_height"], self.input["max_image_width"])
        return decode_stave_band(
            image_bytes, self.input["stave_height"], self.input["stave_width"], self.input["max_stave_width"]
        )

    def predict(self, images: list[torch.Tensor]) -> list[str]:
        """
        LMX transcriptions of prepared stave images, in one batch padded with white as in training. A batch that runs
        out of GPU memory is split in halves, so that only a single stave that does not fit fails.
        """
        try:
            widths = torch.tensor([image.shape[-1] for image in images], dtype=torch.long)
            max_width = int(widths.max())
            batch = torch.stack([F.pad(image, (0, max_width - image.shape[-1]), value=1.0) for image in images])
            generated = self.model.generate(batch.to(self.device), widths)
            return [self.vocab.decode_prediction(ids) for ids in generated.cpu().tolist()]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(images) == 1:
                raise
            print(f"Out of GPU memory with {len(images)} staves; splitting the batch.", file=sys.stderr)
            half = len(images) // 2
            return self.predict(images[:half]) + self.predict(images[half:])


def limit_cuda_memory(device: torch.device, max_memory_gb: float):
    """
    Caps the memory PyTorch's allocator may take on the GPU; going over it raises an out-of-memory error instead.
    The CUDA context itself (a few hundred MB) is not counted.
    """
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = max_memory_gb * 1024**3 / total
    if not 0 < fraction <= 1:
        raise SystemExit(f"--max-memory {max_memory_gb} GB is not within the GPU's {total / 1024**3:.1f} GB.")
    torch.cuda.set_per_process_memory_fraction(fraction, device)


def write_transcription(lmx: str, staff_dir: Path):
    """Writes the LMX and the MusicXML decoded from it; LMX the decoder cannot make sense of is logged by it."""
    (staff_dir / "transcription.lmx").write_text(lmx + "\n", encoding="utf-8")

    # The lmx Decoder crashes on tokens before the first measure (when reporting them), so they are dropped here
    tokens = lmx.split()
    first_measure = tokens.index("measure") if "measure" in tokens else len(tokens)
    if first_measure > 0:
        print(f"Dropping {first_measure} token(s) before the first measure: {' '.join(tokens[:first_measure])}",
              file=sys.stderr)

    decoder = Decoder(errout=sys.stderr)
    decoder.process_text(" ".join(tokens[first_measure:]))
    write_musicxml_tree_to_file(
        file_path=staff_dir / "transcription.musicxml",
        musicxml_tree=part_to_score(decoder.part_element),
        make_parent_folder=False,
    )


# ==============================================================================
# Serve: the Musibot worker IPC
# ==============================================================================

def run_executions(transcriber: Transcriber, executions: list[dict], pages_dir: Path) -> list[tuple[str, str | None]]:
    """
    Transcribes the staves of `executions` in one batch. Returns (execution_id, error) for each, error being None on
    success, so that one unreadable staff fails only its own execution.
    """
    outcomes: dict[str, str | None] = {}
    prepared: list[tuple[str, Path, torch.Tensor]] = []
    for execution in executions:
        execution_id = execution["execution_id"]
        if len(execution["input"]) != 1:
            outcomes[execution_id] = f"Expected one staff image, got {len(execution['input'])} files."
            continue
        image_path = pages_dir / execution["page"] / execution["input"][0]
        try:
            prepared.append((execution_id, image_path.parent, transcriber.prepare(image_path.read_bytes())))
        except (OSError, ValueError) as e:
            outcomes[execution_id] = f"Could not read the staff image '{execution['input'][0]}': {e}"

    if prepared:
        try:
            transcriptions = transcriber.predict([image for _, _, image in prepared])
        except Exception as e:  # e.g. out of GPU memory: the whole batch fails, the model keeps running
            transcriptions = None
            for execution_id, _, _ in prepared:
                outcomes[execution_id] = f"Transcription failed: {e}"
        if transcriptions is not None:
            for (execution_id, staff_dir, _), lmx in zip(prepared, transcriptions, strict=True):
                try:
                    write_transcription(lmx, staff_dir)
                    outcomes[execution_id] = None
                except Exception as e:
                    outcomes[execution_id] = f"Could not convert the transcription to MusicXML: {e}"

    return [(execution["execution_id"], outcomes[execution["execution_id"]]) for execution in executions]


def serve(transcriber: Transcriber, commands: TextIO, results: TextIO, pages_dir: Path, batch_size: int = 16):
    """The worker IPC loop: 'ready' once, then one command at a time until 'shutdown' or EOF."""
    def send(message: dict):
        results.write(json.dumps(message) + "\n")
        results.flush()  # a pipe is block-buffered; without this the Worker Head waits forever

    send({
        "type": "ready",
        "ipc_version": IPC_VERSION,
        "model": {
            "name": transcriber.name,
            "version": transcriber.version,
            "signature": SIGNATURE,
            "supports_batching": True,
        },
    })

    for line in commands:
        command = json.loads(line)
        if command["type"] == "shutdown":
            break
        if command["type"] == "execute":
            executions = [command]
        elif command["type"] == "execute-batch":
            executions = command["executions"]
        else:
            continue  # unknown types are ignored, so that the protocol can grow

        for start in range(0, len(executions), batch_size):
            chunk = executions[start:start + batch_size]
            for execution in chunk:
                print(f"transcribing {execution['page']}/{', '.join(execution['input'])}")
            for execution_id, error in run_executions(transcriber, chunk, pages_dir):
                if error is None:
                    send({"type": "completed", "execution_id": execution_id})
                else:
                    print(error, file=sys.stderr)
                    send({"type": "failed", "execution_id": execution_id, "error": error})


def serve_main(args: argparse.Namespace):
    try:
        command_fd = int(os.environ["MUSIBOT_IPC_COMMAND_FD"])
        result_fd = int(os.environ["MUSIBOT_IPC_RESULT_FD"])
        pages_dir = Path(os.environ["MUSIBOT_PAGES_DIR"])
    except KeyError as e:
        raise SystemExit(f"{e} is not set: a Musibot Model is started by a Worker Head, not by hand.") from e

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.max_memory is not None and device.type == "cuda":
        limit_cuda_memory(device, args.max_memory)
    # Weights are loaded before 'ready', so that the Worker Head offers no work before the model can do it
    transcriber = Transcriber(Path(args.model_snapshot), device, bf16=args.bf16)
    print(f"musibot_model: serving {transcriber.name} {transcriber.version} ({transcriber.encoder_type}) on {device}")

    with os.fdopen(command_fd, "r", encoding="utf-8") as commands, \
         os.fdopen(result_fd, "w", encoding="utf-8") as results:
        serve(transcriber, commands, results, pages_dir, args.batch_size)


def main():
    parser = argparse.ArgumentParser(description="Musibot wrapper of the trained Zeus baseline and MuSViT e2e models.")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Turn a training checkpoint into a model snapshot folder.")
    export.add_argument("checkpoint", help="Training checkpoint, e.g. experiment_results_e2e/musvit_e2e_best.pt.")
    export.add_argument("output", help="Snapshot folder to create, e.g. snapshots/musvit-e2e-2026-10-15.model.")
    export.add_argument("--name", help="Model name announced to Musibot (default: 'zeus-baseline' or 'musvit-e2e').")
    export.add_argument("--version", help="Model version announced to Musibot, what pipelines pin "
                                          "(default: the checkpoint's time and epoch, e.g. 2026-10-15-153411-e180).")
    legacy = export.add_argument_group(
        "checkpoints without a stored configuration",
        "Checkpoints saved before train.py stored the model configuration need the train.py arguments the run "
        "was trained with (defaults: those of train.py). Ignored for newer checkpoints.",
    )
    legacy.add_argument("--max-gen-length", type=int, help="'Max generation length' printed by train.py (required).")
    legacy.add_argument("--timestep-width", type=int)
    legacy.add_argument("--image-height", type=int)
    legacy.add_argument("--max-image-width", type=int)
    legacy.add_argument("--musvit-model", help="Only its configuration is downloaded; the weights are in the checkpoint.")
    legacy.add_argument("--musvit-input", choices=["canvas", "interpolate"])
    legacy.add_argument("--stave-height", type=int)
    legacy.add_argument("--stave-width", help="A width, or 'aspect'.")
    legacy.add_argument("--max-stave-width", type=int)
    legacy.add_argument("--feature-layout", choices=["columns", "raster"])
    legacy.add_argument("--hf-token", help="Hugging Face token for the gated MuSViT configuration (default: HF_TOKEN).")

    serve_parser = commands.add_parser("serve", help="Run a snapshot as a Musibot Model (started by a Worker Head).")
    serve_parser.add_argument("--model-snapshot", required=True, help="Snapshot folder made by 'export'.")
    serve_parser.add_argument("--device", help="'cuda' or 'cpu' (default: cuda if available).")
    serve_parser.add_argument("--bf16", action="store_true", help="Run the MuSViT backbone in bfloat16 autocast (Ampere+ GPUs).")
    serve_parser.add_argument("--batch-size", type=int, default=16, help="Staves per forward pass of an execute-batch.")
    serve_parser.add_argument("--max-memory", type=float, metavar="GB",
                              help="Cap of the GPU memory PyTorch may use, in GB (without the CUDA context). A batch "
                                   "that does not fit is split. Default: no cap.")

    args = parser.parse_args()
    if args.command == "export":
        export_snapshot(args)
    else:
        serve_main(args)


if __name__ == "__main__":
    main()
