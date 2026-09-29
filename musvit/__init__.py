"""
MusViT integration package for OMR experiments.
"""

from musvit.embed_stave import (
    extract_embeddings,
    find_staves,
    get_default_transform,
    load_musvit_model,
    load_stave_image,
)

__all__ = [
    "extract_embeddings",
    "find_staves",
    "get_default_transform",
    "load_musvit_model",
    "load_stave_image",
]
