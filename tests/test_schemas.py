"""
tests/test_schemas.py
Testes unitários de validação e integridade dos schemas Pydantic em app/schemas.py.
"""
import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)


@pytest.mark.unit
class TestPredictRequestSchema:
    def test_default_values(self):
        req = PredictRequest()
        assert req.confidence == 0.25
        assert req.model_name == "yolo-epi.pt"
        assert req.image_base64 is None
        assert req.image_url is None

    def test_valid_confidence_boundaries(self):
        req_min = PredictRequest(confidence=0.0)
        req_max = PredictRequest(confidence=1.0)
        assert req_min.confidence == 0.0
        assert req_max.confidence == 1.0

    def test_invalid_confidence_too_low(self):
        with pytest.raises(ValidationError):
            PredictRequest(confidence=-0.01)

    def test_invalid_confidence_too_high(self):
        with pytest.raises(ValidationError):
            PredictRequest(confidence=1.01)


@pytest.mark.unit
class TestDetectionAndResponseSchema:
    def test_detection_serialization(self):
        det = Detection(label="person", confidence=0.92, bbox=[10.0, 20.0, 100.0, 200.0])
        data = det.model_dump()
        assert data["label"] == "person"
        assert data["confidence"] == 0.92
        assert len(data["bbox"]) == 4

    def test_predict_response_structure(self):
        det = Detection(label="capacete", confidence=0.85, bbox=[0, 0, 50, 50])
        resp = PredictResponse(
            detections=[det],
            inference_ms=15.4,
            model_used="yolo-epi.pt",
            image_width=640,
            image_height=480,
        )
        data = resp.model_dump()
        assert len(data["detections"]) == 1
        assert data["inference_ms"] == 15.4
        assert data["image_width"] == 640


@pytest.mark.unit
class TestBatchSchemas:
    def test_batch_request_validation(self):
        req = BatchPredictRequest(images_base64=["b64_1", "b64_2"], confidence=0.5)
        assert len(req.images_base64) == 2
        assert req.confidence == 0.5
        assert req.model_name == "yolo-epi.pt"

    def test_batch_response_structure(self):
        det = Detection(label="colete", confidence=0.9, bbox=[5, 5, 20, 20])
        pred_resp = PredictResponse(
            detections=[det],
            inference_ms=12.0,
            model_used="yolo-epi.pt",
            image_width=320,
            image_height=320,
        )
        batch_resp = BatchPredictResponse(results=[pred_resp], total_inference_ms=25.0)
        assert len(batch_resp.results) == 1
        assert batch_resp.total_inference_ms == 25.0


@pytest.mark.unit
class TestHealthAndMetricsSchemas:
    def test_health_response(self):
        health = HealthResponse(status="ok", model_loaded=True, model_name="yolov8n.pt")
        assert health.status == "ok"
        assert health.model_loaded is True

    def test_metrics_response(self):
        metrics = MetricsResponse(total_requests=10, successful_requests=8, avg_inference_ms=23.5)
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 8
        assert metrics.avg_inference_ms == 23.5

