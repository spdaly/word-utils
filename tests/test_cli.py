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

    def test_checks_tesseract_availability(self, sample_docx_with_images, tmp_path, monkeypatch):
        """CLI should error if Tesseract not installed."""
        runner = CliRunner()

        # Mock shutil.which to return None
        monkeypatch.setattr("shutil.which", lambda x: None)

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(tmp_path)
        ])

        assert result.exit_code != 0
        assert "tesseract" in result.output.lower()


class TestEngineOption:
    """Test --engine CLI option."""

    def test_default_engine_is_tesseract(self, sample_docx_with_images, tmp_path):
        """Should use Tesseract by default."""
        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir)
        ])

        assert result.exit_code == 0

    def test_engine_gemini_requires_api_key(self, sample_docx_with_images, tmp_path, monkeypatch):
        """Should fail if Gemini selected without API key."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(main, [
            str(sample_docx_with_images),
            "-o", str(output_dir),
            "--engine", "gemini"
        ])

        assert result.exit_code != 0
        assert "API key" in result.output or "API key" in str(result.exception)

    def test_engine_choice_validation(self, tmp_path):
        """Should reject invalid engine choice."""
        runner = CliRunner()

        result = runner.invoke(main, [
            "test.docx",
            "-o", str(tmp_path),
            "--engine", "invalid"
        ])

        assert result.exit_code != 0
        assert "Invalid value" in result.output


class TestMaxImagesOption:
    """Test --max-images CLI option."""

    def test_max_images_passed_to_processor(self, tmp_path, mocker):
        """Should pass max_images to process_document."""
        # Create dummy docx
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        mock_process = mocker.patch('word_ocr.cli.process_document')
        mock_process.return_value = mocker.Mock(
            success=True,
            markdown_path=tmp_path / "test.md",
            image_count=2,
            total_extracted=5
        )
        mocker.patch('word_ocr.cli.check_tesseract', return_value=True)

        from click.testing import CliRunner
        from word_ocr.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            str(docx_path),
            '-o', str(tmp_path),
            '--max-images', '2'
        ])

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs['max_images'] == 2

    def test_verbose_shows_limited_count(self, tmp_path, mocker):
        """Should show 'X of Y images, limited' in verbose mode."""
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        mock_process = mocker.patch('word_ocr.cli.process_document')
        mock_process.return_value = mocker.Mock(
            success=True,
            markdown_path=tmp_path / "test.md",
            image_count=2,
            total_extracted=5
        )
        mocker.patch('word_ocr.cli.check_tesseract', return_value=True)

        from click.testing import CliRunner
        from word_ocr.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            str(docx_path),
            '-o', str(tmp_path),
            '--max-images', '2',
            '-v'
        ])

        assert "2 of 5 images, limited" in result.output
