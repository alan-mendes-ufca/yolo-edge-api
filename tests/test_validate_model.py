"""
tests/test_validate_model.py
Testes unitários para o script de Quality Gate scripts/validate_model.py.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.validate_model import parse_args


@pytest.mark.unit
def test_parse_args_defaults(monkeypatch):
    """Verifica argumentos padrão da CLI do quality gate."""
    monkeypatch.setattr(sys, "argv", ["validate_model.py"])
    args = parse_args()
    assert args.model == "models/yolo-epi.pt"
    assert args.threshold == 0.60
    assert args.dataset == "dataset/epi-detection/data.yaml"


@pytest.mark.unit
def test_parse_args_custom(monkeypatch):
    """Verifica parsing com argumentos customizados."""
    monkeypatch.setattr(sys, "argv", [
        "validate_model.py",
        "--model", "models/custom.pt",
        "--threshold", "0.75",
        "--dataset", "dataset/data.yaml",
    ])
    args = parse_args()
    assert args.model == "models/custom.pt"
    assert args.threshold == 0.75
    assert args.dataset == "dataset/data.yaml"


@pytest.mark.unit
def test_validate_model_missing_model_exits(monkeypatch):
    """Se o arquivo de modelo não existir, main() deve abortar com exit code 1."""
    monkeypatch.setattr(sys, "argv", ["validate_model.py", "--model", "inexistente.pt"])
    from scripts.validate_model import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@pytest.mark.unit
def test_validate_model_below_threshold_fails(monkeypatch, tmp_path):
    """Se o mAP@0.5 for inferior ao limiar, deve abortar com exit code 1."""
    fake_model = tmp_path / "model.pt"
    fake_model.write_text("dummy")

    monkeypatch.setattr(sys, "argv", [
        "validate_model.py",
        "--model", str(fake_model),
        "--threshold", "0.60",
    ])

    mock_metrics = MagicMock()
    mock_metrics.box.map50 = 0.45  # Abaixo de 0.60

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_metrics

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        from scripts.validate_model import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


@pytest.mark.unit
def test_validate_model_above_threshold_succeeds(monkeypatch, tmp_path):
    """Se o mAP@0.5 for maior ou igual ao limiar, deve aprovar com sucesso."""
    fake_model = tmp_path / "model.pt"
    fake_model.write_text("dummy")

    monkeypatch.setattr(sys, "argv", [
        "validate_model.py",
        "--model", str(fake_model),
        "--threshold", "0.60",
    ])

    mock_metrics = MagicMock()
    mock_metrics.box.map50 = 0.72  # Acima de 0.60

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_metrics

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        from scripts.validate_model import main
        # Não deve lançar SystemExit (ou deve finalizar sem erro)
        main()
