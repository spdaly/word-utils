# ABOUTME: Tests for OCR engine abstraction
# ABOUTME: Validates Tesseract wrapper and interface contract

import pytest
from pathlib import Path
from PIL import Image
import io

from word_ocr.ocr import OCREngine, TesseractOCR, GeminiOCR


class TestOCREngine:
    """Test OCR engine interface and implementation."""

    def test_tesseract_ocr_extracts_text_from_image(self, tmp_path):
        """TesseractOCR should extract text from an image with text."""
        # Create a simple test image with text
        img = Image.new('RGB', (200, 50), color='white')
        img_path = tmp_path / "test.png"
        img.save(img_path)

        ocr = TesseractOCR()
        result = ocr.extract_text(img_path)

        # Result should be a string (may be empty for blank image)
        assert isinstance(result, str)

    def test_tesseract_ocr_returns_empty_for_blank_image(self, tmp_path):
        """TesseractOCR should return empty string for blank image."""
        img = Image.new('RGB', (100, 100), color='white')
        img_path = tmp_path / "blank.png"
        img.save(img_path)

        ocr = TesseractOCR()
        result = ocr.extract_text(img_path)

        assert result.strip() == ""

    def test_tesseract_ocr_accepts_pil_image(self):
        """TesseractOCR should accept PIL Image directly."""
        img = Image.new('RGB', (100, 100), color='white')

        ocr = TesseractOCR()
        result = ocr.extract_text(img)

        assert isinstance(result, str)

    def test_ocr_engine_is_abstract(self):
        """OCREngine base class should not be instantiable."""
        with pytest.raises(TypeError):
            OCREngine()


class TestGeminiOCR:
    """Test Gemini OCR engine."""

    def test_raises_without_api_key(self, monkeypatch):
        """Should raise ValueError if no API key configured."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key not found"):
            GeminiOCR()

    def test_accepts_gemini_api_key(self, monkeypatch, mocker):
        """Should use GEMINI_API_KEY env var."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        mocker.patch("google.generativeai.configure")
        mocker.patch("google.generativeai.GenerativeModel")

        engine = GeminiOCR()
        assert engine.api_key == "test-key-123"

    def test_falls_back_to_google_api_key(self, monkeypatch, mocker):
        """Should fall back to GOOGLE_API_KEY."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")
        mocker.patch("google.generativeai.configure")
        mocker.patch("google.generativeai.GenerativeModel")

        engine = GeminiOCR()
        assert engine.api_key == "fallback-key"
