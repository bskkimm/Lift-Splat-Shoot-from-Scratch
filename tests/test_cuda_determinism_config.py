import os

import pytest


def test_cuda_workspace_defaults_when_unset(monkeypatch):
    from lss.reproducibility import configure_cuda_workspace

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: False)
    assert configure_cuda_workspace() == ":4096:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_cuda_workspace_preserves_supported_user_value(monkeypatch):
    from lss.reproducibility import configure_cuda_workspace

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert configure_cuda_workspace() == ":16:8"


def test_cuda_workspace_rejects_unsupported_user_value(monkeypatch):
    from lss.reproducibility import configure_cuda_workspace

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_cuda_workspace()


def test_cuda_workspace_requires_configuration_before_cuda_initialization(monkeypatch):
    from lss.reproducibility import configure_cuda_workspace

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_cuda_workspace()
