"""
tests/conftest.py
Fixtures compartilhadas para a suíte de testes do yolo-edge-api.
Fornece clientes de teste, geradores de dados sintéticos e mocks de hardware/rede.
"""
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import _metrics, app

ASSETS_DIR = Path(__file__).parent / "assets"
ZIDANE_PATH = ASSETS_DIR / "zidane.jpg"


@pytest.fixture(autouse=True)
def reset_api_metrics():
    """Reseta o estado global de métricas antes e depois de cada teste."""
    _metrics["total"] = 0
    _metrics["success"] = 0
    _metrics["total_ms"] = 0.0
    yield
    _metrics["total"] = 0
    _metrics["success"] = 0
    _metrics["total_ms"] = 0.0


@pytest.fixture
def client():
    """Retorna uma instância de TestClient para o FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image_np():
    """Gera um frame RGB sintético de 480x640."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_b64():
    """Gera uma imagem sintética codificada em Base64 (JPEG)."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def sample_zidane_b64():
    """Retorna a imagem de referência zidane.jpg em Base64."""
    if not ZIDANE_PATH.exists():
        pytest.skip("Asset zidane.jpg não encontrado")
    return base64.b64encode(ZIDANE_PATH.read_bytes()).decode("utf-8")


@pytest.fixture
def sample_jpeg_bytes():
    """Retorna bytes brutos de um JPEG sintético."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_yolo_result():
    """Mock de retorno de inferência do Ultralytics YOLO."""
    box_mock = MagicMock()
    box_mock.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([10.0, 20.0, 100.0, 200.0])))))]
    box_mock.cls = [MagicMock(item=MagicMock(return_value=0))]
    box_mock.conf = [MagicMock(item=MagicMock(return_value=0.88))]

    res_mock = MagicMock()
    res_mock.boxes = [box_mock]
    res_mock.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return res_mock

