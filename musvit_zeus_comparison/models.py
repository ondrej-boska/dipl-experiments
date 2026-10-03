"""
Model architectures for Zeus Baseline (Run 1) and MuSViT + Zeus (Run 2).
Shared Bahdanau Attention LSTM Decoder ensures 100% identical decoding mechanics.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self._cached_encoder_proj: torch.Tensor | None = None

    def precompute(self, encoder_outputs: torch.Tensor):
        """Precomputes projected encoder representations to avoid recomputation at every token step."""
        self._cached_encoder_proj = self.encoder_proj(encoder_outputs)

    def clear_cache(self):
        self._cached_encoder_proj = None

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state: (h, c) each of shape (B, decoder_dim)
        prev_state = torch.cat(state, dim=-1)  # (B, 2 * decoder_dim)

        enc_proj = self._cached_encoder_proj if self._cached_encoder_proj is not None else self.encoder_proj(encoder_outputs)
        dec_proj = self.decoder_proj(prev_state).unsqueeze(1)  # (B, 1, attention_dim)

        # Energy scores: (B, T, 1) -> (B, T)
        scores = self.score_proj(torch.tanh(enc_proj + dec_proj)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-1e9"))

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
            context_mask: optional (B, T) mask
        Returns:
            logits: (B, L, vocab_size)
        """
        batch_size, seq_len = target_seq.shape
        embedded = self.dropout(self.embedding(target_seq))  # (B, L, dim)

        # Initialize hidden and cell states with zeros
        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        outputs = []
        self.attention.precompute(context)
        try:
            for t in range(seq_len):
                x_t = embedded[:, t, :]  # (B, dim)
                ctx_t, _ = self.attention(context, (h, c), mask=context_mask)  # (B, dim)
                lstm_input = torch.cat([x_t, ctx_t], dim=-1)  # (B, 2*dim)
                h, c = self.lstm_cell(lstm_input, (h, c))
                outputs.append(h)
        finally:
            self.attention.clear_cache()

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

        curr_token = torch.full((batch_size,), self.bos_idx, dtype=torch.long, device=context.device)
        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        preds = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=context.device)

        self.attention.precompute(context)
        try:
            for _ in range(max_len):
                x_t = self.embedding(curr_token)  # (B, dim)
                ctx_t, _ = self.attention(context, (h, c), mask=context_mask)
                lstm_input = torch.cat([x_t, ctx_t], dim=-1)
                h, c = self.lstm_cell(lstm_input, (h, c))
                logits = self.fc_out(h)  # (B, vocab_size)
                next_token = torch.argmax(logits, dim=-1)  # (B,)

                # If sequence was already finished, record pad_idx
                token_to_record = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)
                preds.append(token_to_record)

                is_finished = is_finished | (next_token == self.eos_idx)
                if is_finished.all():
                    break
                curr_token = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)
        finally:
            self.attention.clear_cache()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, 2 * out_dim)
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
        rnn_layers = []
        for layer_idx in range(num_lstm_layers):
            in_d = self.rnn_input_size if layer_idx == 0 else dim
            rnn_layers.append(
                BiLSTMSumBlock(in_dim=in_d, out_dim=dim, dropout=dropout, residual=(layer_idx > 0))
            )
        self.rnn = nn.Sequential(*rnn_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, 1, H, W)
        feat = self.conv(x)  # (B, C, H', W')
        B, C, H, W = feat.shape

        # Permute to (B, W', H'*C)
        feat = feat.permute(0, 3, 2, 1).contiguous().view(B, W, H * C)

        # Pad horizontal steps if not divisible by remaining
        if self.remaining > 1:
            pad_w = (-W) % self.remaining
            if pad_w > 0:
                feat = F.pad(feat, (0, 0, 0, pad_w))
                W = feat.shape[1]
            feat = feat.reshape(B, W // self.remaining, H * C * self.remaining)

        feat = self.pre_rnn_dropout(feat)
        out = self.rnn(feat)  # (B, W // remaining, dim)
        return out, None


# ==============================================================================
# Run 2: Minimal MuSViT Encoder (Pre-extracted Features or Live Backbone)
# ==============================================================================

class MusvitEncoder(nn.Module):
    """
    Minimal MuSViT Encoder (Run 2).
    Adapts pre-extracted MuSViT representations (or on-the-fly ViT features)
    to match the exact dimensional and sequential expectations of ZeusDecoder.
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

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: (B, T, 768) pre-extracted MuSViT features (e.g. T=64 after vertical pooling)
            mask: optional attention mask
        Returns:
            context: (B, T, dim) ready for ZeusDecoder
        """
        # Linear projection: (B, T, 768) -> (B, T, dim)
        projected = self.proj(features)

        # Sequential contextualization: (B, T, dim) -> (B, T, dim)
        context = self.lstm(projected)
        context = self.layer_norm(context + projected)  # Residual connection

        return context, mask


# ==============================================================================
# Combined End-to-End OMR Model
# ==============================================================================

class CombinedOMRModel(nn.Module):
    """
    Unified container wrapping either:
    - Run 1: ZeusEncoder + ZeusDecoder
    - Run 2: MusvitEncoder + ZeusDecoder
    """
    def __init__(
        self,
        encoder_type: Literal["zeus", "musvit"],
        vocab_size: int,
        dim: int = 256,
        timestep_width: int = 16,
        bos_idx: int = 0,
        eos_idx: int = 1,
        pad_idx: int = 2,
        max_length: int = 600,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "zeus":
            self.encoder = ZeusEncoder(dim=dim, timestep_width=timestep_width, dropout=dropout)
        elif self.encoder_type == "musvit":
            self.encoder = MusvitEncoder(dim=dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'. Must be 'zeus' or 'musvit'.")

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

    def forward(self, x: torch.Tensor, target_seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x, mask)
        logits = self.decoder(context, target_seq, context_mask=ctx_mask)
        return logits

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x)
        preds = self.decoder.generate(context, max_length=max_length, context_mask=ctx_mask)
        return preds


# Alias for backward compatibility and flexible importing
OMRModel = CombinedOMRModel
