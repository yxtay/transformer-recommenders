import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:1"
# https://docs.pytorch.org/docs/stable/mps_environment_variables.html#mps-environment-variables
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

"""Package-level initialisation for xfmr_rec.

This module sets a few process-wide environment variables that are
recommended for stable behavior with tokenizers and PyTorch memory
management on supported platforms.
"""
