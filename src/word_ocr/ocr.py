# ABOUTME: OCR engine abstraction with Tesseract implementation
# ABOUTME: Provides pluggable interface for text extraction from images

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from PIL import Image
import pytesseract


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
