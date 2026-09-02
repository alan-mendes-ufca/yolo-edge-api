"""
tests/test_gpu.py
Verificação do ambiente de execução do PyTorch, suporte a hardware (CUDA/CPU)
e integridade de compatibilidade de carga de pesos.
"""
import pytest
import torch

import app.model as app_model


@pytest.mark.unit
def test_torch_installation_and_tensor_creation():
    """Verifica se o PyTorch está instalado e executa operações básicas em tensores."""
    tensor_a = torch.ones((2, 2))
    tensor_b = torch.ones((2, 2))
    result = tensor_a + tensor_b
    assert result.shape == (2, 2)
    assert float(result.sum().item()) == 8.0


@pytest.mark.unit
def test_device_detection_consistency():
    """Verifica detecção de CUDA/CPU sem quebrar em máquinas sem GPU dedicada."""
    cuda_available = torch.cuda.is_available()
    assert isinstance(cuda_available, bool)
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        assert isinstance(device_name, str)
        assert len(device_name) > 0
    else:
        assert torch.cuda.device_count() == 0


@pytest.mark.unit
def test_patched_torch_load_weights_only(monkeypatch):
    """Verifica se o patch _patched_torch_load injeta weights_only=False por padrão."""
    captured_kwargs = {}

    def spy_load(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"loaded": True}

    monkeypatch.setattr(app_model, "_orig_torch_load", spy_load)
    app_model._patched_torch_load("dummy.pt")
    assert captured_kwargs.get("weights_only") is False

    # Verifica também quando weights_only é explicitamente passado
    captured_kwargs.clear()
    app_model._patched_torch_load("dummy.pt", weights_only=True)
    assert captured_kwargs.get("weights_only") is True