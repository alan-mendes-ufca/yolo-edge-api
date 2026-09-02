"""
tests/test_model.py
Testes unitários do carregamento, cache e ciclo de vida do modelo em app/model.py.
"""
from unittest.mock import patch

import pytest

import app.model as app_model
from app.model import get_default_model_name, load_model


@pytest.mark.unit
@pytest.mark.model
def test_get_default_model_name_default(monkeypatch):
    """Retorna yolo-epi.pt quando MODEL_NAME não está definida."""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert get_default_model_name() == "yolo-epi.pt"


@pytest.mark.unit
@pytest.mark.model
def test_get_default_model_name_custom_env(monkeypatch):
    """Retorna o valor customizado da variável de ambiente MODEL_NAME."""
    monkeypatch.setenv("MODEL_NAME", "custom-model.pt")
    assert get_default_model_name() == "custom-model.pt"


@pytest.mark.unit
@pytest.mark.model
def test_load_model_file_not_found():
    """Deve levantar FileNotFoundError quando o arquivo de pesos não existe."""
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_model("modelo_inexistente_12345.pt")


@pytest.mark.unit
@pytest.mark.model
def test_load_model_caching(tmp_path, monkeypatch):
    """Garante que load_model armazena e reutiliza o modelo em cache sem recarregá-lo."""
    fake_model_file = tmp_path / "fake_yolo.pt"
    fake_model_file.write_text("fake binary")

    monkeypatch.setattr(app_model, "MODELS_DIR", tmp_path)

    # Limpa cache para teste isolado
    app_model._cache.pop("fake_yolo.pt", None)

    with patch("app.model.YOLO") as mock_yolo_cls:
        mock_instance = mock_yolo_cls.return_value

        # Primeira chamada: instancia YOLO
        model1 = load_model("fake_yolo.pt")
        assert model1 is mock_instance
        assert mock_yolo_cls.call_count == 1

        # Segunda chamada: deve vir do cache (_cache)
        model2 = load_model("fake_yolo.pt")
        assert model2 is mock_instance
        assert mock_yolo_cls.call_count == 1  # não chamou de novo
