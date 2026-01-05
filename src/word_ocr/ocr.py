# ABOUTME: OCR engine abstraction with Tesseract and Gemini implementations
# ABOUTME: Provides pluggable interface for text extraction from images

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from PIL import Image
import pytesseract
from google import genai


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: Union[Path, Image.Image]) -> str:
        """Extract text from an image.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Extracted text as string
        """
        pass


class TesseractOCR(OCREngine):
    """Tesseract-based OCR implementation."""

    def extract_text(self, image: Union[Path, Image.Image]) -> str:
        """Extract text using Tesseract OCR.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Extracted text as string
        """
        if isinstance(image, Path):
            image = Image.open(image)

        text = pytesseract.image_to_string(image)
        return text


class GeminiOCR(OCREngine):
    """Google Gemini-based OCR implementation."""

    DEFAULT_MODEL = "gemini-1.5-flash"
    PROMPT = "Extract all text from this image exactly as it appears. Return only the extracted text, no commentary."

    def __init__(self, model: str = None):
        """Initialize Gemini OCR engine.

        Args:
            model: Gemini model to use (default: gemini-1.5-flash)

        Raises:
            ValueError: If no API key found in environment
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)

    def extract_text(self, image: Union[Path, Image.Image]) -> str:
        """Extract text using Gemini vision model.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Extracted text as string
        """
        if isinstance(image, Path):
            image = Image.open(image)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.PROMPT, image]
        )
        return response.text
