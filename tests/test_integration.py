# ABOUTME: End-to-end integration tests
# ABOUTME: Validates complete workflow from CLI to output

import pytest
from pathlib import Path
from click.testing import CliRunner

from word_ocr.cli import main


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, sample_docx_with_images, tmp_path):
        """Test complete workflow: CLI -> extraction -> OCR -> Markdown."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir),
            "-v"
        ])

        assert result.exit_code == 0

        # Check output structure
        md_file = output_dir / "test_with_images.md"
        images_dir = output_dir / "test_with_images_images"

        assert md_file.exists()
        assert images_dir.exists()

        # Check images were saved
        images = list(images_dir.glob("*.png"))
        assert len(images) == 2

        # Check markdown content
        content = md_file.read_text()
        assert "# test_with_images.docx" in content
        assert "## Image 1" in content
        assert "## Image 2" in content
        assert "image_001.png" in content
        assert "image_002.png" in content

    def test_full_workflow_no_images(self, sample_docx_no_images, tmp_path):
        """Test workflow with document that has no images."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_no_images),
            "-o", str(output_dir),
            "-v"
        ])

        assert result.exit_code == 0

        # Check output exists
        md_file = output_dir / "test_no_images.md"
        assert md_file.exists()

        # Check markdown indicates no images
        content = md_file.read_text()
        assert "# test_no_images.docx" in content
        assert "No images found" in content

    def test_batch_workflow(
        self, sample_docx_with_images, sample_docx_no_images, tmp_path
    ):
        """Test batch processing of multiple documents."""
        import shutil

        runner = CliRunner()

        # Setup batch directory with multiple docs
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        shutil.copy(sample_docx_with_images, batch_dir / "report1.docx")
        shutil.copy(sample_docx_no_images, batch_dir / "report2.docx")

        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(batch_dir / "*.docx"),
            "-o", str(output_dir),
            "-v"
        ])

        assert result.exit_code == 0

        # Both documents processed
        assert (output_dir / "report1.md").exists()
        assert (output_dir / "report2.md").exists()

        # Images extracted only from document with images
        assert (output_dir / "report1_images").exists()
        images = list((output_dir / "report1_images").glob("*.png"))
        assert len(images) == 2

        # Summary in output
        assert "2 succeeded" in result.output
