"""
MusViT and Zeus Comparison Module.

This package provides a unified benchmark environment to compare:
- Run 1 (Zeus Baseline): CNN-BiLSTM Encoder + Zeus Bahdanau Attention Decoder
- Run 2 (MuSViT + Zeus): MuSViT Vision Transformer Feature Extractor + Zeus Decoder
- Run 3 (end-to-end MuSViT + Zeus): MuSViT backbone inside the model (frozen or fine-tuned) + Zeus Decoder

Both runs share the exact same decoder, loss, optimizer, and evaluation metrics.
"""

from .models import (
    ZeusEncoder,
    MusvitEncoder,
    MusvitBackbone,
    MusvitE2EEncoder,
    ZeusDecoder,
    CombinedOMRModel,
    OMRModel,
)

__all__ = [
    "ZeusEncoder",
    "MusvitEncoder",
    "MusvitBackbone",
    "MusvitE2EEncoder",
    "ZeusDecoder",
    "CombinedOMRModel",
    "OMRModel",
]

