# ABOUTME: Tests for Word document image extraction
# ABOUTME: Validates extraction of embedded images from .docx files

import pytest
from pathlib import Path
from PIL import Image

from word_ocr.extractor import ImageExtractor, ExtractedImage


class TestImageExtractor:
    """Test Word document image extraction."""

    def test_extracts_images_from_docx(self, sample_docx_with_images):
        """Extractor should find all images in a Word document."""
        extractor = ImageExtractor()
        images = extractor.extract(sample_docx_with_images)

        assert len(images) == 2
        assert all(isinstance(img, ExtractedImage) for img in images)

    def test_extracted_image_has_pil_image(self, sample_docx_with_images):
        """ExtractedImage should contain a PIL Image."""
        extractor = ImageExtractor()
        images = extractor.extract(sample_docx_with_images)

        assert all(isinstance(img.image, Image.Image) for img in images)

    def test_extracted_image_has_index(self, sample_docx_with_images):
        """ExtractedImage should have sequential index."""
        extractor = ImageExtractor()
        images = extractor.extract(sample_docx_with_images)

        assert images[0].index == 1
        assert images[1].index == 2

    def test_returns_empty_list_for_no_images(self, sample_docx_no_images):
        """Extractor should return empty list when no images."""
        extractor = ImageExtractor()
        images = extractor.extract(sample_docx_no_images)

        assert images == []

    def test_returns_empty_list_for_empty_doc(self, empty_docx):
        """Extractor should return empty list for empty document."""
        extractor = ImageExtractor()
        images = extractor.extract(empty_docx)

        assert images == []

    def test_raises_for_invalid_file(self, tmp_path):
        """Extractor should raise for non-existent file."""
        extractor = ImageExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract(tmp_path / "nonexistent.docx")
