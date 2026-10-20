"""
Benchmarks of OMR models on Musicorpus datasets.

A benchmark compares a gold dataset with a predicted one, page by page: both are folders of Musicorpus page folders,
paired by page ID, and within a page by file path. The predicted dataset typically comes from running a Musibot
model or pipeline over the gold pages (`omr-benchmark predict`).
"""

__version__ = "0.1.0"
