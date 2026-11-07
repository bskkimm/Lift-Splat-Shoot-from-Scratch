import random
import os
import torch


def configure_cuda_workspace():
    """Configure a supported cuBLAS deterministic workspace before CUDA use."""
    key = "CUBLAS_WORKSPACE_CONFIG"
    value = os.environ.get(key)
    if value is None:
        if torch.cuda.is_initialized():
            raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
        value = ":4096:8"
        os.environ[key] = value
    if value not in (":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    return value


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
