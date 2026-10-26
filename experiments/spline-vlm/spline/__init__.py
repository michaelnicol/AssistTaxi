"""Lane-spline training. lanes, data, model, fit, and score each do one job."""

import os

# The Trainer splits each batch over every visible GPU, which changes the math, so
# runs use one GPU unless CUDA_VISIBLE_DEVICES says otherwise. Must run before torch
# touches CUDA, which is why it lives in the package import.
PINNED_GPU = "CUDA_VISIBLE_DEVICES" not in os.environ
if PINNED_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
