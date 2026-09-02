"""
tests/test_api.py
Cobertura completa da YOLO Inference API:
- Smoke tests (/health, /metrics)
- Unit tests (_decode_image, _load_image_from_request)
- Endpoints de inferência REST (/predict via Base64 e URL, /predict/image)
- Endpoints de câmera física (/predict/camera, /predict/camera/image)
- Endpoints de streaming de vídeo (/stream/camera, /stream/view)
- Processamento em lote (/predict/batch)
- Observabilidade e métricas acumuladas
"""
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from app.main import _decode_image, _load_image_from_request
from app.schemas import PredictRequest

ASSETS = Path(__file__).parent / "assets"
DEFAULT_MODEL = "yolov8n.pt"


# ────────────────────────────────────────────────────────────
# SMOKE TESTS
# ────────────────────────────────────────────────────────────

@pytest.mark.smoke
class TestSmoke:
    def test_health_status_200(self, client):
        """API deve retornar HTTP 200 com status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_payload_structure(self, client):
        """Payload deve conter status, model_loaded e model_name."""
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data

    def test_metrics_endpoint_accessible(self, client):
        """Endpoint /metrics deve estar acessível."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "avg_inference_ms" in data


# ────────────────────────────────────────────────────────────
# UNIT TESTS — Funções internas de manipulação de imagem
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDecodeImage:
    def _make_b64_image(self, width=32, height=32, fmt="JPEG"):
        img = Image.new("RGB", (width, height), color=(128, 64, 192))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def test_returns_numpy_array(self):
        result = _decode_image(self._make_b64_image())
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        result = _decode_image(self._make_b64_image(64, 48))
        assert result.shape == (48, 64, 3)

    def test_png_format(self):
        result = _decode_image(self._make_b64_image(fmt="PNG"))
        assert result.shape[2] == 3

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError):
            _decode_image("dado_invalido_nao_e_base64")


@pytest.mark.unit
class TestLoadImageFromRequest:
    def test_missing_both_sources_raises_422(self):
        req = PredictRequest(image_base64=None, image_url=None)
        with pytest.raises(HTTPException) as exc_info:
            _load_image_from_request(req)
        assert exc_info.value.status_code == 422

    def test_load_from_base64(self, sample_image_b64):
        req = PredictRequest(image_base64=sample_image_b64)
        img_np = _load_image_from_request(req)
        assert isinstance(img_np, np.ndarray)
        assert img_np.shape == (64, 64, 3)

    def test_load_from_url_success(self, sample_jpeg_bytes):
        req = PredictRequest(image_url="http://fake-cdn.local/image.jpg")
        mock_resp = MagicMock()
        mock_resp.content = sample_jpeg_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            img_np = _load_image_from_request(req)
            assert isinstance(img_np, np.ndarray)
            assert img_np.shape == (32, 32, 3)

    def test_load_from_url_failure(self):
        req = PredictRequest(image_url="http://invalid.local/missing.jpg")
        with (
            patch("httpx.get", side_effect=httpx.ConnectError("Conexão falhou")),
            pytest.raises(httpx.ConnectError),
        ):
            _load_image_from_request(req)


# ────────────────────────────────────────────────────────────
# INTEGRATION TESTS — Endpoint /predict (Base64 e URL)
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestPredictEndpoint:
    def test_predict_returns_200(self, client, sample_zidane_b64):
        resp = client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        })
        assert resp.status_code == 200

    def test_predict_detects_at_least_one_object(self, client, sample_zidane_b64):
        data = client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        }).json()
        assert len(data["detections"]) >= 1

    def test_predict_response_schema(self, client, sample_zidane_b64):
        data = client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        }).json()
        assert "detections" in data
        assert "inference_ms" in data
        assert "model_used" in data
        assert "image_width" in data
        assert "image_height" in data
        assert data["inference_ms"] > 0

    def test_predict_detection_fields(self, client, sample_zidane_b64):
        data = client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        }).json()
        for det in data["detections"]:
            assert isinstance(det["label"], str)
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4

    def test_predict_missing_input_returns_422(self, client):
        resp = client.post("/predict", json={
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        })
        assert resp.status_code == 422

    def test_predict_model_not_found_returns_404(self, client, sample_image_b64):
        resp = client.post("/predict", json={
            "image_base64": sample_image_b64,
            "model_name": "inexistente.pt",
        })
        assert resp.status_code == 404

    def test_predict_with_image_url_success(self, client, sample_jpeg_bytes):
        mock_resp = MagicMock()
        mock_resp.content = sample_jpeg_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            resp = client.post("/predict", json={
                "image_url": "http://fake-cdn.local/foto.jpg",
                "model_name": DEFAULT_MODEL,
                "confidence": 0.25,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["image_width"] == 32
            assert data["image_height"] == 32


# ────────────────────────────────────────────────────────────
# ENDPOINT /predict/image — Renderização visual anotada
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestPredictImageEndpoint:
    def test_predict_image_returns_jpeg(self, client, sample_zidane_b64):
        resp = client.post("/predict/image", json={
            "image_base64": sample_zidane_b64,
            "model_name": DEFAULT_MODEL,
            "confidence": 0.3,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        # Valida que o payload binário é um JPEG decodificável
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "JPEG"

    def test_predict_image_missing_input_returns_422(self, client):
        resp = client.post("/predict/image", json={
            "model_name": DEFAULT_MODEL,
        })
        assert resp.status_code == 422

    def test_predict_image_model_not_found_returns_404(self, client, sample_image_b64):
        resp = client.post("/predict/image", json={
            "image_base64": sample_image_b64,
            "model_name": "modelo_que_nao_existe.pt",
        })
        assert resp.status_code == 404


# ────────────────────────────────────────────────────────────
# ENDPOINTS DE CÂMERA FÍSICA (/predict/camera, /predict/camera/image)
# ────────────────────────────────────────────────────────────

@pytest.mark.camera
@pytest.mark.unit
class TestCameraEndpoints:
    @patch("app.main._capture_frame_from_camera")
    def test_predict_camera_success(self, mock_capture, client, sample_image_np):
        mock_capture.return_value = sample_image_np
        resp = client.post(f"/predict/camera?model_name={DEFAULT_MODEL}&confidence=0.25")
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert data["model_used"] == DEFAULT_MODEL
        assert data["image_width"] == 640
        assert data["image_height"] == 480

    @patch("app.main._capture_frame_from_camera")
    def test_predict_camera_hardware_error_returns_500(self, mock_capture, client):
        mock_capture.side_effect = HTTPException(status_code=500, detail="Falha ao capturar imagem da câmera.")
        resp = client.post("/predict/camera")
        assert resp.status_code == 500
        assert "câmera" in resp.json()["detail"]

    @patch("app.main._capture_frame_from_camera")
    def test_predict_camera_image_success(self, mock_capture, client, sample_image_np):
        mock_capture.return_value = sample_image_np
        resp = client.get(f"/predict/camera/image?model_name={DEFAULT_MODEL}&confidence=0.25")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "JPEG"

    @patch("app.main._capture_frame_from_camera")
    def test_predict_camera_image_hardware_error(self, mock_capture, client):
        mock_capture.side_effect = HTTPException(status_code=500, detail="Falha no dispositivo de vídeo.")
        resp = client.get("/predict/camera/image")
        assert resp.status_code == 500


# ────────────────────────────────────────────────────────────
# STREAMING DE VÍDEO MJPEG (/stream/view, /stream/camera)
# ────────────────────────────────────────────────────────────

@pytest.mark.camera
@pytest.mark.unit
class TestStreamEndpoints:
    def test_stream_view_returns_html(self, client):
        resp = client.get("/stream/view")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert 'src="/stream/camera"' in resp.text

    @patch("app.main._streaming_lock.locked", return_value=True)
    def test_stream_camera_conflict_when_locked(self, mock_locked, client):
        """Se o streaming_lock já estiver adquirido, deve responder HTTP 409 Conflict."""
        resp = client.get(f"/stream/camera?model_name={DEFAULT_MODEL}")
        assert resp.status_code == 409
        assert "Já existe um stream de câmera" in resp.json()["detail"]

    @patch("app.main.load_model")
    @patch("subprocess.Popen")
    def test_stream_camera_success_headers(self, mock_popen, mock_load, client):
        """Verifica se o endpoint responde com Content-Type multipart/x-mixed-replace."""
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_proc.stdout.read = MagicMock(return_value=b"")
        mock_proc.stderr = None
        mock_popen.return_value = mock_proc

        resp = client.get(f"/stream/camera?model_name={DEFAULT_MODEL}")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers.get("content-type", "")


# ────────────────────────────────────────────────────────────
# BATCH ENDPOINT (/predict/batch)
# ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestBatchEndpoint:
    @pytest.fixture
    def two_images_b64(self, sample_zidane_b64):
        return [sample_zidane_b64, sample_zidane_b64]

    def test_batch_returns_correct_count(self, client, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        }).json()
        assert len(data["results"]) == 2

    def test_batch_total_ms_is_positive(self, client, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
            "model_name": DEFAULT_MODEL,
        }).json()
        assert data["total_inference_ms"] > 0


# ────────────────────────────────────────────────────────────
# OBSERVABILIDADE E MÉTRICAS ACUMULADAS (/metrics)
# ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetricsTracking:
    def test_metrics_increment_on_requests(self, client, sample_zidane_b64):
        initial = client.get("/metrics").json()
        assert initial["total_requests"] == 0
        assert initial["successful_requests"] == 0

        # Faz 2 requisições bem-sucedidas
        client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "model_name": DEFAULT_MODEL,
        })
        client.post("/predict", json={
            "image_base64": sample_zidane_b64,
            "model_name": DEFAULT_MODEL,
        })

        after = client.get("/metrics").json()
        assert after["total_requests"] >= 2
        assert after["successful_requests"] >= 2
        assert after["avg_inference_ms"] > 0
