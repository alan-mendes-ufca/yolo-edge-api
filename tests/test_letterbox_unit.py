"""
tests/test_letterbox_unit.py
Testes unitários dedicados às transformações geométricas de letterbox e ajuste inverso de bboxes.
"""
import numpy as np
import pytest

from preprocessing.utils.letterbox import adjust_bboxes, letterbox


@pytest.mark.unit
@pytest.mark.preprocessor
class TestLetterboxTransform:
    def test_square_image_no_padding(self):
        """Imagem já quadrada no tamanho alvo não recebe padding."""
        img = np.full((320, 320, 3), 50, dtype=np.uint8)
        out, scale, (pad_w, pad_h) = letterbox(img, target_size=320)
        assert out.shape == (320, 320, 3)
        assert scale == 1.0
        assert pad_w == 0
        assert pad_h == 0

    def test_wide_image_vertical_padding(self):
        """Imagem panorâmica (largura > altura) deve receber padding vertical no topo/base."""
        img = np.full((240, 640, 3), 80, dtype=np.uint8)
        out, scale, (pad_w, pad_h) = letterbox(img, target_size=640)
        assert out.shape == (640, 640, 3)
        assert scale == 1.0  # 640 / 640
        assert pad_w == 0
        assert pad_h == (640 - 240) // 2  # 200

    def test_tall_image_horizontal_padding(self):
        """Imagem vertical (altura > largura) deve receber padding horizontal nas laterais."""
        img = np.full((640, 320, 3), 80, dtype=np.uint8)
        out, scale, (pad_w, pad_h) = letterbox(img, target_size=640)
        assert out.shape == (640, 640, 3)
        assert scale == 1.0
        assert pad_h == 0
        assert pad_w == (640 - 320) // 2  # 160

    def test_padding_color_value(self):
        """O padding adicionado deve ter o valor especificado (padrão 114)."""
        img = np.full((200, 400, 3), 255, dtype=np.uint8)
        out, _, (_pad_w, _pad_h) = letterbox(img, target_size=400, pad_color=114)
        # O canto superior esquerdo deve ser padding
        assert np.all(out[0, 0] == 114)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestAdjustBboxes:
    def test_roundtrip_coordinate_mapping(self):
        """Verifica se mapear uma bbox original para letterbox e de volta reproduz as coordenadas originais."""
        orig_h, orig_w = 480, 640
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        target_size = 640

        _, scale, (pad_w, pad_h) = letterbox(img, target_size=target_size)

        # Bbox no espaço original: [x1, y1, x2, y2]
        orig_bbox = np.array([[50.0, 60.0, 200.0, 300.0]], dtype=float)

        # Mapeia manualmente para espaço letterboxed
        lb_bbox = orig_bbox * scale
        lb_bbox[:, [0, 2]] += pad_w
        lb_bbox[:, [1, 3]] += pad_h

        # Executa a função adjust_bboxes para reverter
        recovered = adjust_bboxes(lb_bbox, scale=scale, pad_w=pad_w, pad_h=pad_h)

        np.testing.assert_allclose(recovered, orig_bbox, atol=1e-5)
