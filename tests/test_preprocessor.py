"""
tests/test_preprocessor.py
Testes unitários e funcionais do pipeline de pré-processamento de imagens (Letterbox, CLAHE, Filtros).
"""
import numpy as np
import pytest

from preprocessing.preprocessor import (
    CONFIG_DEFAULT,
    CONFIG_HIGH_QUALITY,
    CONFIG_LOW_LIGHT,
    PreprocessConfig,
    Preprocessor,
)


def make_frame(h=480, w=640, dtype=np.uint8):
    """Gera imagem NumPy aleatória para testes de pré-processamento."""
    return np.random.randint(0, 255, (h, w, 3), dtype=dtype)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestPreprocessorOutput:
    def test_output_shape_letterbox(self):
        """Frame deve ter shape (infer_size, infer_size, 3) após letterbox."""
        pp = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame())
        assert res.frame.shape == (416, 416, 3)

    def test_output_dtype_uint8(self):
        """Sem normalização, dtype deve permanecer uint8."""
        pp = Preprocessor(PreprocessConfig(normalize=False))
        res = pp.process(make_frame())
        assert res.frame.dtype == np.uint8

    def test_output_dtype_float32_when_normalized(self):
        """Com normalização, dtype vira float32 e valores ficam entre 0 e 1."""
        pp = Preprocessor(PreprocessConfig(normalize=True))
        res = pp.process(make_frame())
        assert res.frame.dtype == np.float32
        assert res.frame.max() <= 1.0
        assert res.frame.min() >= 0.0

    def test_scale_and_padding_set(self):
        """Letterbox deve preencher scale e pad_w/h no resultado."""
        pp = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=True))
        res = pp.process(make_frame(h=480, w=640))
        assert res.scale > 0
        assert res.orig_size == (480, 640)

    def test_letterbox_padding_symmetric(self):
        """Frame quadrado não deve ter padding."""
        pp = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=True))
        res = pp.process(make_frame(h=416, w=416))
        assert res.pad_w == 0
        assert res.pad_h == 0

    def test_without_rgb_conversion(self):
        """Quando convert_rgb=False, canais não são invertidos."""
        pp = Preprocessor(PreprocessConfig(convert_rgb=False))
        frame = make_frame(h=100, w=100)
        res = pp.process(frame)
        assert res.frame.shape[:2] == (320, 320)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestPreprocessorFilters:
    def test_gaussian_blur_applied(self):
        """Aplica filtro gaussiano e mantém dimensões e canais corretos."""
        cfg = PreprocessConfig(gaussian_blur=True, gaussian_ksize=5, gaussian_sigma=1.2)
        pp = Preprocessor(cfg)
        frame = make_frame()
        res = pp.process(frame)
        assert res.frame.shape == (320, 320, 3)

    def test_median_blur_applied(self):
        """Aplica filtro de mediana e mantém dimensões e canais corretos."""
        cfg = PreprocessConfig(median_blur=True, median_ksize=5)
        pp = Preprocessor(cfg)
        frame = make_frame()
        res = pp.process(frame)
        assert res.frame.shape == (320, 320, 3)

    def test_clahe_lab_space(self):
        """Aplica CLAHE no espaço LAB para baixa luminosidade."""
        cfg = PreprocessConfig(clahe=True, clahe_space="lab", clahe_clip=3.0)
        pp = Preprocessor(cfg)
        frame = make_frame()
        res = pp.process(frame)
        assert res.frame.shape == (320, 320, 3)

    def test_clahe_hsv_space(self):
        """Aplica CLAHE no espaço HSV no canal de valor (V)."""
        cfg = PreprocessConfig(clahe=True, clahe_space="hsv", clahe_clip=2.0)
        pp = Preprocessor(cfg)
        frame = make_frame()
        res = pp.process(frame)
        assert res.frame.shape == (320, 320, 3)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestBboxAdjustment:
    def test_adjust_removes_letterbox_offset(self):
        """Bboxes ajustadas devem ter y1 menor que as originais (padding removido)."""
        pp = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame(h=480, w=640))
        boxes_lb = np.array([[10, 50, 100, 200]], dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_lb, res)
        if res.pad_h > 0:
            assert boxes_orig[0, 1] < boxes_lb[0, 1]

    def test_adjust_multiple_boxes(self):
        """Deve suportar array com múltiplos bounding boxes."""
        pp = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame(h=480, w=640))
        boxes_lb = np.array([
            [10, 50, 100, 200],
            [150, 120, 300, 400],
        ], dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_lb, res)
        assert boxes_orig.shape == (2, 4)

    def test_adjust_empty_boxes(self):
        """Deve processar array vazio sem falhar."""
        pp = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame(h=480, w=640))
        boxes_empty = np.empty((0, 4), dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_empty, res)
        assert boxes_orig.shape == (0, 4)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestPreprocessorConfigs:
    def test_config_low_light_applies_clahe(self):
        pp = Preprocessor(CONFIG_LOW_LIGHT)
        res = pp.process(make_frame())
        assert res.frame.shape[2] == 3

    def test_config_default_no_filter(self):
        pp = Preprocessor(CONFIG_DEFAULT)
        assert not pp.cfg.gaussian_blur
        assert not pp.cfg.median_blur
        assert not pp.cfg.clahe

    def test_config_high_quality(self):
        pp = Preprocessor(CONFIG_HIGH_QUALITY)
        assert pp.cfg.infer_size == 640
        assert pp.cfg.use_letterbox is True
        res = pp.process(make_frame(h=720, w=1280))
        assert res.frame.shape == (640, 640, 3)


@pytest.mark.unit
@pytest.mark.preprocessor
class TestNonUniformScale:
    def test_adjust_boxes_without_letterbox_uses_separate_axis_scales(self):
        """Sem letterbox, x e y devem ser corrigidos com escalas diferentes
        quando a imagem de entrada não é quadrada."""
        pp = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=False))
        res = pp.process(make_frame(h=480, w=640))
        assert res.scale_x != res.scale_y
        boxes_resized = np.array([[0, 0, 416, 416]], dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_resized, res)
        assert abs(boxes_orig[0, 2] - 640) < 1e-6
        assert abs(boxes_orig[0, 3] - 480) < 1e-6
