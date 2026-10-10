"""
Model architectures for Zeus Baseline (Run 1), MuSViT + Zeus on cached features (Run 2)
and end-to-end MuSViT + Zeus with an optionally fine-tuned backbone (Run 3).
Shared Bahdanau Attention LSTM Decoder ensures 100% identical decoding mechanics.
"""

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ViTModel
except ImportError:
    ViTModel = None


def lengths_to_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool mask on `device`, True at valid timesteps."""
    lengths = lengths.to(device, non_blocking=True)
    return torch.arange(max_len, device=device)[None, :] < lengths[:, None]


# ==============================================================================
# Shared Bahdanau Attention & LSTM Decoder
# ==============================================================================

class BahdanauAttention(nn.Module):
    """
    Standard additive Bahdanau Attention matching Zeus specification (rnn_cells_with_attention.py).
    Computes attention scores over encoder outputs given previous decoder state.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=True)
        self.decoder_proj = nn.Linear(decoder_dim * 2, attention_dim, bias=True)  # concatenated (h, c)
        self.score_proj = nn.Linear(attention_dim, 1, bias=True)

    def project_encoder(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Projects encoder outputs once per sequence, instead of at every decoding step."""
        return self.encoder_proj(encoder_outputs)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_proj: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state: (h, c) each of shape (B, decoder_dim)
        prev_state = torch.cat(state, dim=-1)  # (B, 2 * decoder_dim)
        dec_proj = self.decoder_proj(prev_state).unsqueeze(1)  # (B, 1, attention_dim)

        # Energy scores: (B, T, 1) -> (B, T)
        scores = self.score_proj(torch.tanh(encoder_proj + dec_proj)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)

        return context, attn_weights


class ZeusDecoder(nn.Module):
    """
    Autoregressive single-layer LSTM decoder with Bahdanau attention matching Zeus specification.
    Zeus decoder has no dropout by default; all model dropout is localized in the encoder.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        bos_idx: int = 0,
        eos_idx: int = 1,
        pad_idx: int = 2,
        max_length: int = 600,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_dim=dim, decoder_dim=dim, attention_dim=dim)
        self.lstm_cell = nn.LSTMCell(input_size=dim * 2, hidden_size=dim)  # input: [embedded_token; context]
        self.fc_out = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        context: torch.Tensor,
        target_seq: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training matching Zeus decoder_training.
        Args:
            context: (B, T, dim) encoder output sequence
            target_seq: (B, L) input token IDs (starting with BOS)
            context_mask: optional (B, T) mask, True at valid timesteps
        Returns:
            logits: (B, L, vocab_size)
        """
        batch_size, seq_len = target_seq.shape
        embedded = self.dropout(self.embedding(target_seq))  # (B, L, dim)
        encoder_proj = self.attention.project_encoder(context)

        # Initialize hidden and cell states with zeros
        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (B, dim)
            ctx_t, _ = self.attention(context, encoder_proj, (h, c), mask=context_mask)  # (B, dim)
            lstm_input = torch.cat([x_t, ctx_t], dim=-1)  # (B, 2*dim)
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs.append(h)

        stacked_h = torch.stack(outputs, dim=1)  # (B, L, dim)
        logits = self.fc_out(self.dropout(stacked_h))  # (B, L, vocab_size)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        context: torch.Tensor,
        max_length: int | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding for evaluation / prediction.
        Returns:
            predicted_tokens: (B, max_len)
        """
        max_len = max_length or self.max_length
        batch_size = context.shape[0]
        encoder_proj = self.attention.project_encoder(context)

        curr_token = torch.full((batch_size,), self.bos_idx, dtype=torch.long, device=context.device)
        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        preds = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=context.device)

        for step in range(max_len):
            x_t = self.embedding(curr_token)  # (B, dim)
            ctx_t, _ = self.attention(context, encoder_proj, (h, c), mask=context_mask)
            lstm_input = torch.cat([x_t, ctx_t], dim=-1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.fc_out(h)  # (B, vocab_size)
            next_token = torch.argmax(logits, dim=-1)  # (B,)

            # If sequence was already finished, record pad_idx
            token_to_record = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)
            preds.append(token_to_record)

            is_finished = is_finished | (next_token == self.eos_idx)
            # Checking on the CPU waits for the GPU, so do it only every few steps; extra steps only add padding
            if step % 8 == 7 and is_finished.all():
                break
            curr_token = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)

        if not preds:
            return torch.empty((batch_size, 0), dtype=torch.long, device=context.device)

        return torch.stack(preds, dim=1)  # (B, pred_len)


# ==============================================================================
# Run 1: Zeus CNN-BiLSTM Encoder
# ==============================================================================

class ResNetConvBlock(nn.Module):
    """Residual convolutional block matching Zeus architecture."""
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class BiLSTMSumBlock(nn.Module):
    """
    Bidirectional LSTM block with merge_mode='sum' and optional residual connection.
    Replicates the exact sequence modeling in Zeus Keras (keras_model.py:105-111).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, residual: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = residual

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            out, _ = self.lstm(x)  # (B, T, 2 * out_dim)
        else:
            # Packing keeps the backward direction from reading the batch padding first.
            # It needs the lengths on the CPU; keeping them there avoids waiting for the GPU.
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        fwd, bwd = out.chunk(2, dim=-1)
        summed = self.dropout(fwd + bwd)  # (B, T, out_dim)
        if self.residual:
            summed = summed + x
        return summed


class ZeusEncoder(nn.Module):
    """
    Standard Zeus CNN-BiLSTM Encoder (Run 1 Baseline).
    Reduces 2D stave image to 1D horizontal time sequence.
    Follows solo26 architecture with timestep_width reduction and residual BiLSTM layers.
    """
    def __init__(
        self,
        dim: int = 256,
        in_channels: int = 1,
        cnn_ch: int = 32,
        cnn_stages: int = 3,  # 3 stages for single-staff (downsampling = 2^3 = 8)
        input_height: int = 96,
        timestep_width: int = 16,  # Matches solo26.yaml (16 // 8 = 2x horizontal reduction)
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.cnn_stages = cnn_stages
        self.input_height = input_height
        self.timestep_width = timestep_width

        self.remaining = timestep_width // (2 ** cnn_stages)
        if self.remaining < 1:
            raise ValueError(
                f"Inconsistent settings of timestep_width ({timestep_width}) "
                f"and cnn_stages ({cnn_stages}): timestep_width must be >= {2 ** cnn_stages}"
            )

        # Initial Conv layer without BatchNorm or ReLU, matching Zeus keras_model.py:45-47
        layers = [
            nn.Conv2d(in_channels, cnn_ch, kernel_size=3, stride=1, padding=1, bias=False),
        ]

        curr_ch = cnn_ch
        for i in range(cnn_stages):
            out_ch = min(dim, cnn_ch * (2 ** i))
            layers.append(ResNetConvBlock(curr_ch, out_ch, downsample=True))
            layers.append(ResNetConvBlock(out_ch, out_ch, downsample=False))
            curr_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pre_rnn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        reduced_h = input_height // (2 ** cnn_stages)
        self.rnn_input_size = curr_ch * reduced_h * self.remaining

        # Bidirectional LSTM layers with merge_mode='sum' and residual connection on layer 1+
        self.rnn = nn.ModuleList(
            BiLSTMSumBlock(
                in_dim=self.rnn_input_size if layer_idx == 0 else dim,
                out_dim=dim,
                dropout=dropout,
                residual=(layer_idx > 0),
            )
            for layer_idx in range(num_lstm_layers)
        )

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Number of encoder timesteps produced from images of the given unpadded widths."""
        lengths = widths
        for _ in range(self.cnn_stages):
            lengths = (lengths - 1) // 2 + 1  # 3x3 conv, stride 2, padding 1
        return (lengths + self.remaining - 1) // self.remaining

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, 1, H, W), lengths: optional (B,) unpadded image widths, preferably on the CPU
        feat = self.conv(x)  # (B, C, H', W')
        B, C, H, W = feat.shape

        # Permute to (B, W', H'*C)
        feat = feat.permute(0, 3, 2, 1).reshape(B, W, H * C)

        # Pad horizontal steps if not divisible by remaining
        if self.remaining > 1:
            pad_w = (-W) % self.remaining
            if pad_w > 0:
                feat = F.pad(feat, (0, 0, 0, pad_w))
                W = feat.shape[1]
            feat = feat.reshape(B, W // self.remaining, H * C * self.remaining)

        out = self.pre_rnn_dropout(feat)
        steps = self.output_lengths(lengths) if lengths is not None else None
        for layer in self.rnn:
            out = layer(out, steps)  # (B, W // remaining, dim)

        mask = lengths_to_mask(steps, out.shape[1], out.device) if steps is not None else None
        return out, mask


# ==============================================================================
# Run 2: Minimal MuSViT Encoder (Pre-extracted Features)
# ==============================================================================

class MusvitEncoder(nn.Module):
    """
    Minimal MuSViT Encoder (Run 2).
    Adapts pre-extracted MuSViT representations to match
    the exact dimensional and sequential expectations of ZeusDecoder.
    """
    def __init__(
        self,
        dim: int = 256,
        musvit_dim: int = 768,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.musvit_dim = musvit_dim

        # Projection layer from MuSViT embedding space to model dimension
        self.proj = nn.Sequential(
            nn.Linear(musvit_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

        # Contextualizer BiLSTM matching Zeus's merge_mode='sum' sequence modeling
        self.lstm = BiLSTMSumBlock(in_dim=dim, out_dim=dim, dropout=dropout, residual=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: (B, T, musvit_dim) pre-extracted MuSViT features
                (e.g. T=64 patch columns with 4 rows of 768 concatenated, musvit_dim=3072)
            lengths: optional (B,) unpadded sequence lengths, preferably on the CPU
        Returns:
            context: (B, T, dim) ready for ZeusDecoder, and the (B, T) mask of valid timesteps
        """
        # Linear projection: (B, T, musvit_dim) -> (B, T, dim)
        projected = self.proj(features)

        # Sequential contextualization: (B, T, dim) -> (B, T, dim)
        context = self.lstm(projected, lengths)
        context = self.layer_norm(context + projected)  # Residual connection

        mask = lengths_to_mask(lengths, context.shape[1], context.device) if lengths is not None else None
        return context, mask


# ==============================================================================
# MuSViT Backbone (shared by feature extraction and Run 3)
# ==============================================================================

def load_musvit(model_name: str = "PRAIG/musvit", device: str | None = None, token: str | None = None):
    """
    Loads a MuSViT encoder (e.g. 'PRAIG/musvit' or 'PRAIG/musvit-light') as a plain ViTModel, as its model card
    prescribes: AutoModel would give ViT-MAE, which randomly masks 70% of the patches.
    """
    if ViTModel is None:
        raise ImportError("Please install transformers: pip install transformers")

    print(f"Loading pre-trained MuSViT model '{model_name}'...")
    hf_token = token or os.environ.get("HF_TOKEN")
    # The MAE checkpoint has no pooler, and the pooler is unused here
    model, loading_info = ViTModel.from_pretrained(
        model_name, token=hf_token, add_pooling_layer=False, output_loading_info=True
    )

    # Anything missing would be randomly initialized
    missing = [k for k in loading_info["missing_keys"] if not k.startswith("pooler.")]
    if missing or loading_info["mismatched_keys"]:
        raise RuntimeError(
            f"'{model_name}' did not load cleanly into ViTModel. "
            f"Missing weights: {missing}. Mismatched weights: {loading_info['mismatched_keys']}."
        )

    if device is not None:
        model.to(device)
    model.eval()
    return model


def stave_canvas(bands: torch.Tensor, canvas_size: int = 1024) -> torch.Tensor:
    """
    MuSViT's documented stave input: the stave bands (B, 3, stave_height, canvas_size) pasted at the top
    of a white square canvas of the size MuSViT was pre-trained on -> (B, 3, canvas_size, canvas_size).
    Built on the bands' device, so that only the bands travel from the CPU.
    """
    canvas = bands.new_ones(bands.shape[0], 3, canvas_size, canvas_size)  # white, as to_tensor maps 255 to 1.0
    canvas[:, :, :bands.shape[2]] = bands
    return canvas


def stave_patch_grid(last_hidden_state: torch.Tensor, patch_rows: int, patch_columns: int) -> torch.Tensor:
    """
    Cuts the stave's features out of the ViT output.

    Args:
        last_hidden_state: (B, 1 + N, D) with the [CLS] token first and N patches in row-major order
        patch_rows: number of top patch rows covered by the stave
        patch_columns: number of patch columns of the ViT input
    Returns:
        (B, patch_rows, patch_columns, D) patch features of the stave, without any white padding rows below it
    """
    patches = last_hidden_state[:, 1:, :]  # drop [CLS]
    B, N, D = patches.shape
    if N % patch_columns != 0 or N // patch_columns < patch_rows:
        raise ValueError(f"Expected at least {patch_rows} rows of {patch_columns} patch tokens, got {N} tokens.")
    return patches.reshape(B, N // patch_columns, patch_columns, D)[:, :patch_rows]


def arrange_patch_grid(grid: torch.Tensor, layout: str) -> torch.Tensor:
    """
    Arranges a (..., rows, columns, D) MuSViT patch grid as a sequence:
    - 'columns': one timestep per patch column, its rows concatenated -> (..., columns, rows * D)
    - 'raster': row-major patch sequence, as in the MuSViT documentation -> (..., rows * columns, D)
    """
    if layout == "columns":
        # Like the Zeus encoder flattening H x C per column, keeps the vertical (pitch) position
        return grid.transpose(-3, -2).flatten(-2)
    if layout == "raster":
        return grid.flatten(-3, -2)
    raise ValueError(f"Unknown feature layout '{layout}'. Must be 'columns' or 'raster'.")


class MusvitBackbone(nn.Module):
    """
    Pre-trained MuSViT (or another ViT on the Hugging Face Hub) applied to stave bands, trainable or frozen.

    Input modes:
    - 'canvas': the band is pasted on the white square canvas MuSViT was pre-trained on, exactly as
      extract_features.py does; the ViT processes all canvas_size^2 / patch_size^2 patches (4096 for MuSViT).
    - 'interpolate': the band alone is processed with interpolated position embeddings, which the MuSViT model
      card recommends for fine-tuning; 16x fewer patches at height 64, so far cheaper to fine-tune.

    While frozen, it runs without gradients and stays in eval mode even when the model is trained.
    """
    def __init__(
        self,
        model_name: str = "PRAIG/musvit",
        stave_height: int = 64,
        input_mode: Literal["canvas", "interpolate"] = "canvas",
        token: str | None = None,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if input_mode not in ("canvas", "interpolate"):
            raise ValueError(f"Unknown input_mode '{input_mode}'. Must be 'canvas' or 'interpolate'.")
        self.vit = load_musvit(model_name, token=token)
        self.model_name = model_name
        self.input_mode = input_mode
        self.bf16 = bf16
        self.canvas_size = self.vit.config.image_size
        self.patch_size = self.vit.config.patch_size
        self.hidden_size = self.vit.config.hidden_size

        if stave_height % self.patch_size != 0 or not 0 < stave_height <= self.canvas_size:
            raise ValueError(
                f"The stave height must be a multiple of the patch size ({self.patch_size}) "
                f"between {self.patch_size} and {self.canvas_size}, got {stave_height}."
            )
        self.stave_height = stave_height
        self.patch_rows = stave_height // self.patch_size

        if gradient_checkpointing:
            # Takes effect only while the ViT is in training mode, i.e. while fine-tuning
            self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.trainable = False
        self.set_trainable(False)

    def set_trainable(self, trainable: bool):
        """Freezes or unfreezes the ViT weights."""
        self.trainable = trainable
        self.vit.requires_grad_(trainable)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.vit.eval()
        return self

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bands: (B, 3, stave_height, width) stave bands in [0, 1], without normalization (as documented);
                the width must be the canvas size in 'canvas' mode and a multiple of the patch size otherwise
        Returns:
            (B, patch_rows, width / patch_size, hidden_size) patch grid of the staves, in FP32
        """
        height, width = bands.shape[-2:]
        if height != self.stave_height or width % self.patch_size != 0:
            raise ValueError(
                f"Expected stave bands of height {self.stave_height} and a width divisible by {self.patch_size}, "
                f"got {height}x{width}."
            )
        if self.input_mode == "canvas" and width != self.canvas_size:
            raise ValueError(f"The 'canvas' input mode needs stave bands {self.canvas_size} px wide, got {width}.")

        with (
            torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()),
            torch.autocast(bands.device.type, dtype=torch.bfloat16, enabled=self.bf16),
        ):
            if self.input_mode == "canvas":
                hidden = self.vit(stave_canvas(bands, self.canvas_size)).last_hidden_state
            else:
                hidden = self.vit(bands, interpolate_pos_encoding=True).last_hidden_state
        return stave_patch_grid(hidden.float(), self.patch_rows, width // self.patch_size)


# ==============================================================================
# Run 3: End-to-End MuSViT Encoder (Backbone + Run 2 Adapter)
# ==============================================================================

class MusvitE2EEncoder(MusvitEncoder):
    """
    MuSViT backbone followed by the Run 2 adapter, taking stave bands instead of cached features.
    Its state dict is a superset of MusvitEncoder's, so a Run 2 checkpoint initializes the adapter
    (and the decoder) when the feature settings match, while the backbone keeps its pre-trained weights.
    """
    def __init__(
        self,
        backbone: MusvitBackbone,
        dim: int = 256,
        feature_layout: Literal["columns", "raster"] = "columns",
        dropout: float = 0.2,
    ):
        musvit_dim = backbone.hidden_size * (backbone.patch_rows if feature_layout == "columns" else 1)
        super().__init__(dim=dim, musvit_dim=musvit_dim, dropout=dropout)
        self.backbone = backbone
        self.feature_layout = feature_layout

    def forward(self, bands: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            bands: (B, 3, stave_height, width) stave bands in [0, 1]
            lengths: ignored; all bands of a batch have the same width, so no timestep is padding
        """
        features = arrange_patch_grid(self.backbone(bands), self.feature_layout)
        return super().forward(features)


# ==============================================================================
# Combined End-to-End OMR Model
# ==============================================================================

class CombinedOMRModel(nn.Module):
    """
    Unified container wrapping either:
    - Run 1: ZeusEncoder + ZeusDecoder
    - Run 2: MusvitEncoder + ZeusDecoder (pre-extracted features)
    - Run 3: MusvitE2EEncoder + ZeusDecoder (MuSViT backbone in the model, frozen or fine-tuned)
    """
    def __init__(
        self,
        encoder_type: Literal["zeus", "musvit", "musvit_e2e"],
        vocab_size: int,
        dim: int = 256,
        timestep_width: int = 16,
        input_height: int = 96,
        musvit_dim: int = 768,
        bos_idx: int = 0,
        eos_idx: int = 1,
        pad_idx: int = 2,
        max_length: int = 600,
        dropout: float = 0.2,
        musvit_model: str = "PRAIG/musvit",
        musvit_input: Literal["canvas", "interpolate"] = "canvas",
        stave_height: int = 64,
        feature_layout: Literal["columns", "raster"] = "columns",
        hf_token: str | None = None,
        musvit_bf16: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "zeus":
            self.encoder = ZeusEncoder(dim=dim, input_height=input_height, timestep_width=timestep_width, dropout=dropout)
        elif self.encoder_type == "musvit":
            self.encoder = MusvitEncoder(dim=dim, musvit_dim=musvit_dim, dropout=dropout)
        elif self.encoder_type == "musvit_e2e":
            backbone = MusvitBackbone(
                model_name=musvit_model,
                stave_height=stave_height,
                input_mode=musvit_input,
                token=hf_token,
                bf16=musvit_bf16,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.encoder = MusvitE2EEncoder(backbone, dim=dim, feature_layout=feature_layout, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'. Must be 'zeus', 'musvit' or 'musvit_e2e'.")

        # In Zeus, all model dropout is localized in the encoder; decoder has 0 dropout
        self.decoder = ZeusDecoder(
            vocab_size=vocab_size,
            dim=dim,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            max_length=max_length,
            dropout=0.0,
        )

    @property
    def backbone(self) -> MusvitBackbone | None:
        """The pre-trained MuSViT backbone of Run 3, or None."""
        return getattr(self.encoder, "backbone", None)

    def forward(self, x: torch.Tensor, target_seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x, lengths)
        logits = self.decoder(context, target_seq, context_mask=ctx_mask)
        return logits

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, lengths: torch.Tensor | None = None, max_length: int | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x, lengths)
        preds = self.decoder.generate(context, max_length=max_length, context_mask=ctx_mask)
        return preds


# Alias for backward compatibility and flexible importing
OMRModel = CombinedOMRModel
