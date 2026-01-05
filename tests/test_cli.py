# ABOUTME: Tests for CLI interface
# ABOUTME: Validates command-line argument handling and output

import pytest
from pathlib import Path
from click.testing import CliRunner

from word_ocr.cli import main


class TestCLI:
    """Test CLI interface."""

    def test_processes_single_file(self, sample_docx_with_images, tmp_path):
        """CLI should process a single file."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir)
        ])

        assert result.exit_code == 0
        assert (output_dir / "test_with_images.md").exists()

    def test_processes_glob_pattern(
        self, sample_docx_with_images, sample_docx_no_images, tmp_path
    ):
        """CLI should process files matching glob pattern."""
        runner = CliRunner()

        # Setup batch directory
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        import shutil
        shutil.copy(sample_docx_with_images, batch_dir / "doc1.docx")
        shutil.copy(sample_docx_no_images, batch_dir / "doc2.docx")

        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(batch_dir / "*.docx"),
            "-o", str(output_dir)
        ])

        assert result.exit_code == 0
        assert (output_dir / "doc1.md").exists()
        assert (output_dir / "doc2.md").exists()

    def test_dry_run_lists_files(self, sample_docx_with_images, tmp_path):
        """CLI --dry-run should list files without processing."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir),
            "--dry-run"
        ])

        assert result.exit_code == 0
        assert "test_with_images.docx" in result.output
        assert not (output_dir / "test_with_images.md").exists()

    def test_verbose_output(self, sample_docx_with_images, tmp_path):
        """CLI -v should show progress."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir),
            "-v"
        ])

        assert result.exit_code == 0
        assert "Processing" in result.output

    def test_exit_code_1_on_partial_failure(self, sample_docx_with_images, tmp_path):
        """CLI should exit 1 when some files fail."""
        runner = CliRunner()

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        import shutil
        shutil.copy(sample_docx_with_images, batch_dir / "good.docx")
        (batch_dir / "bad.docx").write_text("not valid")

        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(batch_dir / "*.docx"),
            "-o", str(output_dir)
        ])

        assert result.exit_code == 1

    def test_exit_code_2_on_complete_failure(self, tmp_path):
        """CLI should exit 2 when all files fail."""
        runner = CliRunner()

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        (batch_dir / "bad.docx").write_text("not valid")

        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(batch_dir / "*.docx"),
            "-o", str(output_dir)
        ])

        assert result.exit_code == 2

    def test_checks_tesseract_availability(self, tmp_path, monkeypatch):
        """CLI should error if Tesseract not installed."""
        runner = CliRunner()

        # Mock shutil.which to return None
        monkeypatch.setattr("shutil.which", lambda x: None)

        result = runner.invoke(main, [
            "test.docx",
            "-o", str(tmp_path)
        ])

        assert result.exit_code != 0
        assert "tesseract" in result.output.lower()
