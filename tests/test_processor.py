# ABOUTME: Integration tests for document processing pipeline
# ABOUTME: Tests full flow from .docx to Markdown output

import pytest
from pathlib import Path

from word_ocr.processor import process_document, process_batch, ProcessResult


class TestProcessDocument:
    """Test single document processing."""

    def test_processes_document_with_images(self, sample_docx_with_images, tmp_path):
        """Should process document and create output."""
        output_dir = tmp_path / "output"

        result = process_document(
            input_path=sample_docx_with_images,
            output_dir=output_dir
        )

        assert isinstance(result, ProcessResult)
        assert result.success is True
        assert result.markdown_path.exists()
        assert result.images_dir.exists()
        assert result.image_count == 2
        assert result.error is None

    def test_processes_document_without_images(self, sample_docx_no_images, tmp_path):
        """Should handle document with no images."""
        output_dir = tmp_path / "output"

        result = process_document(
            input_path=sample_docx_no_images,
            output_dir=output_dir
        )

        assert result.success is True
        assert result.markdown_path.exists()
        assert result.image_count == 0

    def test_fails_gracefully_for_missing_file(self, tmp_path):
        """Should return failure result for missing file."""
        result = process_document(
            input_path=tmp_path / "missing.docx",
            output_dir=tmp_path / "output"
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestProcessBatch:
    """Test batch document processing."""

    def test_processes_multiple_documents(
        self, sample_docx_with_images, sample_docx_no_images, tmp_path
    ):
        """Should process multiple documents."""
        # Copy files to a batch directory
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        import shutil
        shutil.copy(sample_docx_with_images, batch_dir / "doc1.docx")
        shutil.copy(sample_docx_no_images, batch_dir / "doc2.docx")

        output_dir = tmp_path / "output"

        results = process_batch(
            input_pattern=str(batch_dir / "*.docx"),
            output_dir=output_dir
        )

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_continues_on_failure(self, sample_docx_with_images, tmp_path):
        """Should continue processing after one failure."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        import shutil
        shutil.copy(sample_docx_with_images, batch_dir / "good.docx")

        # Create invalid file
        (batch_dir / "bad.docx").write_text("not a docx")

        output_dir = tmp_path / "output"

        results = process_batch(
            input_pattern=str(batch_dir / "*.docx"),
            output_dir=output_dir
        )

        assert len(results) == 2
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1
