# ABOUTME: Tests for Markdown output generation
# ABOUTME: Validates correct formatting and file structure

import pytest
from pathlib import Path
from PIL import Image

from word_ocr.renderer import MarkdownRenderer, ProcessedImage


class TestMarkdownRenderer:
    """Test Markdown output generation."""

    def test_renders_single_image(self, tmp_path):
        """Renderer should create markdown with one image."""
        renderer = MarkdownRenderer()
        output_dir = tmp_path / "output"

        images = [
            ProcessedImage(
                image=Image.new('RGB', (100, 100), 'red'),
                index=1,
                ocr_text="Hello world"
            )
        ]

        result = renderer.render(
            images=images,
            source_name="test.docx",
            output_dir=output_dir
        )

        assert result.markdown_path.exists()
        assert result.images_dir.exists()
        assert (result.images_dir / "image_001.png").exists()

        content = result.markdown_path.read_text()
        assert "# test.docx" in content
        assert "![Image 1](test_images/image_001.png)" in content
        assert "Hello world" in content

    def test_renders_multiple_images(self, tmp_path):
        """Renderer should handle multiple images."""
        renderer = MarkdownRenderer()
        output_dir = tmp_path / "output"

        images = [
            ProcessedImage(
                image=Image.new('RGB', (100, 100), 'red'),
                index=1,
                ocr_text="First"
            ),
            ProcessedImage(
                image=Image.new('RGB', (100, 100), 'blue'),
                index=2,
                ocr_text="Second"
            )
        ]

        result = renderer.render(
            images=images,
            source_name="multi.docx",
            output_dir=output_dir
        )

        content = result.markdown_path.read_text()
        assert "## Image 1" in content
        assert "## Image 2" in content
        assert "First" in content
        assert "Second" in content

    def test_renders_empty_document(self, tmp_path):
        """Renderer should handle document with no images."""
        renderer = MarkdownRenderer()
        output_dir = tmp_path / "output"

        result = renderer.render(
            images=[],
            source_name="empty.docx",
            output_dir=output_dir
        )

        content = result.markdown_path.read_text()
        assert "# empty.docx" in content
        assert "No images found" in content

    def test_handles_ocr_failure(self, tmp_path):
        """Renderer should handle OCR failure placeholder."""
        renderer = MarkdownRenderer()
        output_dir = tmp_path / "output"

        images = [
            ProcessedImage(
                image=Image.new('RGB', (100, 100), 'red'),
                index=1,
                ocr_text=None  # OCR failed
            )
        ]

        result = renderer.render(
            images=images,
            source_name="test.docx",
            output_dir=output_dir
        )

        content = result.markdown_path.read_text()
        assert "[OCR failed for this image]" in content
