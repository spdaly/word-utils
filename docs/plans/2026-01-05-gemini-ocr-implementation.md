# Gemini OCR Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GeminiOCR engine that uses Google's Gemini vision model for improved OCR accuracy.

**Architecture:** New `GeminiOCR` class implementing existing `OCREngine` interface. API key via environment variable, configurable model (default flash, option pro). CLI gets `--engine` flag to select OCR engine.

**Tech Stack:** google-generativeai SDK, PIL for image handling, click for CLI

---

## Task 1: Add Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add google-generativeai to dependencies**

```toml
# In pyproject.toml, update dependencies list:
dependencies = [
    "python-docx>=1.1.0",
    "pytesseract>=0.3.10",
    "Pillow>=10.0.0",
    "click>=8.1.0",
    "google-generativeai>=0.5.0",
]
```

**Step 2: Install updated dependencies**

Run: `uv pip install -e ".[dev]"`
Expected: Successfully installed google-generativeai

**Step 3: Verify installation**

Run: `uv run python -c "import google.generativeai; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add google-generativeai dependency"
```

---

## Task 2: GeminiOCR Class - API Key Validation

**Files:**
- Modify: `src/word_ocr/ocr.py`
- Modify: `tests/test_ocr.py`

**Step 1: Write failing tests for API key validation**

```python
# tests/test_ocr.py - add to existing file

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
```

**Step 2: Add pytest-mock to dev dependencies**

```toml
# pyproject.toml - update dev dependencies
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.0.0", "pytest-mock>=3.12.0"]
```

Run: `uv pip install -e ".[dev]"`

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr.py::TestGeminiOCR -v`
Expected: FAIL with `ImportError: cannot import name 'GeminiOCR'`

**Step 4: Write minimal implementation**

```python
# src/word_ocr/ocr.py - add after TesseractOCR class

import os
import google.generativeai as genai


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
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def extract_text(self, image: Union[Path, Image.Image]) -> str:
        """Extract text using Gemini vision model.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Extracted text as string
        """
        if isinstance(image, Path):
            image = Image.open(image)

        response = self.client.generate_content([self.PROMPT, image])
        return response.text
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr.py::TestGeminiOCR -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/word_ocr/ocr.py tests/test_ocr.py pyproject.toml
git commit -m "feat: add GeminiOCR class with API key validation"
```

---

## Task 3: GeminiOCR Class - Model Configuration

**Files:**
- Modify: `tests/test_ocr.py`

**Step 1: Write failing tests for model configuration**

```python
# tests/test_ocr.py - add to TestGeminiOCR class

    def test_default_model_is_flash(self, monkeypatch, mocker):
        """Should default to gemini-1.5-flash."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mocker.patch("google.generativeai.configure")
        mock_model = mocker.patch("google.generativeai.GenerativeModel")

        engine = GeminiOCR()

        assert engine.model == "gemini-1.5-flash"
        mock_model.assert_called_once_with("gemini-1.5-flash")

    def test_uses_custom_model(self, monkeypatch, mocker):
        """Should use specified model."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mocker.patch("google.generativeai.configure")
        mock_model = mocker.patch("google.generativeai.GenerativeModel")

        engine = GeminiOCR(model="gemini-1.5-pro")

        assert engine.model == "gemini-1.5-pro"
        mock_model.assert_called_once_with("gemini-1.5-pro")
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr.py::TestGeminiOCR -v`
Expected: PASS (5 tests) - implementation already supports this

**Step 3: Commit**

```bash
git add tests/test_ocr.py
git commit -m "test: add model configuration tests for GeminiOCR"
```

---

## Task 4: GeminiOCR Class - Text Extraction

**Files:**
- Modify: `tests/test_ocr.py`

**Step 1: Write test for text extraction**

```python
# tests/test_ocr.py - add to TestGeminiOCR class

    def test_extracts_text_from_pil_image(self, monkeypatch, mocker):
        """Should extract text from PIL Image."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mocker.patch("google.generativeai.configure")

        mock_response = mocker.Mock()
        mock_response.text = "Extracted text from image"
        mock_client = mocker.Mock()
        mock_client.generate_content.return_value = mock_response
        mocker.patch("google.generativeai.GenerativeModel", return_value=mock_client)

        engine = GeminiOCR()
        image = Image.new('RGB', (100, 100), color='white')
        result = engine.extract_text(image)

        assert result == "Extracted text from image"
        mock_client.generate_content.assert_called_once()
        call_args = mock_client.generate_content.call_args[0][0]
        assert call_args[0] == GeminiOCR.PROMPT
        assert call_args[1] == image

    def test_extracts_text_from_path(self, monkeypatch, mocker, tmp_path):
        """Should extract text from image path."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mocker.patch("google.generativeai.configure")

        mock_response = mocker.Mock()
        mock_response.text = "Text from file"
        mock_client = mocker.Mock()
        mock_client.generate_content.return_value = mock_response
        mocker.patch("google.generativeai.GenerativeModel", return_value=mock_client)

        # Create test image file
        img_path = tmp_path / "test.png"
        Image.new('RGB', (100, 100), color='blue').save(img_path)

        engine = GeminiOCR()
        result = engine.extract_text(img_path)

        assert result == "Text from file"
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr.py::TestGeminiOCR -v`
Expected: PASS (7 tests)

**Step 3: Commit**

```bash
git add tests/test_ocr.py
git commit -m "test: add text extraction tests for GeminiOCR"
```

---

## Task 5: Export GeminiOCR

**Files:**
- Modify: `src/word_ocr/__init__.py`

**Step 1: Update exports**

```python
# src/word_ocr/__init__.py
# ABOUTME: Word OCR public API exports
# ABOUTME: Exposes process_document and process_batch functions

from .processor import process_document, process_batch, ProcessResult
from .ocr import OCREngine, TesseractOCR, GeminiOCR
from .extractor import ImageExtractor, ExtractedImage
from .renderer import MarkdownRenderer, ProcessedImage, RenderResult

__all__ = [
    "process_document",
    "process_batch",
    "ProcessResult",
    "OCREngine",
    "TesseractOCR",
    "GeminiOCR",
    "ImageExtractor",
    "ExtractedImage",
    "MarkdownRenderer",
    "ProcessedImage",
    "RenderResult",
]
```

**Step 2: Verify import works**

Run: `uv run python -c "from word_ocr import GeminiOCR; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/word_ocr/__init__.py
git commit -m "feat: export GeminiOCR from package"
```

---

## Task 6: CLI --engine Option

**Files:**
- Modify: `src/word_ocr/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests for --engine option**

```python
# tests/test_cli.py - add new test class

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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::TestEngineOption -v`
Expected: FAIL with `Error: No such option: --engine`

**Step 3: Update CLI implementation**

```python
# src/word_ocr/cli.py - update imports and main function

# Add to imports at top:
from .ocr import TesseractOCR, GeminiOCR

# Update main function signature - add engine option after dry_run:
@click.option(
    "--engine",
    type=click.Choice(["tesseract", "gemini", "gemini-pro"]),
    default="tesseract",
    help="OCR engine to use (default: tesseract)"
)
def main(input_pattern: str, output: str, verbose: bool, dry_run: bool, engine: str):

# Replace the Tesseract check and add engine selection after dry_run handling:
    # Select OCR engine
    if engine == "tesseract":
        if not check_tesseract():
            click.echo("Error: Tesseract is not installed.", err=True)
            click.echo("Install with: brew install tesseract (macOS)", err=True)
            click.echo("             apt install tesseract-ocr (Linux)", err=True)
            sys.exit(2)
        ocr_engine = TesseractOCR()
    elif engine == "gemini":
        ocr_engine = GeminiOCR()
    elif engine == "gemini-pro":
        ocr_engine = GeminiOCR(model="gemini-1.5-pro")

# Update process_document call to use ocr_engine:
        result = process_document(
            input_path=file_path,
            output_dir=output_dir,
            ocr_engine=ocr_engine
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::TestEngineOption -v`
Expected: PASS (3 tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (should be ~39 tests now)

**Step 6: Commit**

```bash
git add src/word_ocr/cli.py tests/test_cli.py
git commit -m "feat: add --engine CLI option for OCR engine selection"
```

---

## Task 7: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add OCR Engines section to README**

```markdown
# Add after "Exit codes:" section and before "### Library"

### OCR Engines

**Tesseract (default)** - Local processing, free, requires Tesseract installed

```bash
word-ocr document.docx -o ./output
word-ocr document.docx -o ./output --engine tesseract
```

**Gemini** - Cloud-based, better accuracy for complex documents, requires API key

```bash
export GEMINI_API_KEY="your-api-key"
word-ocr document.docx -o ./output --engine gemini
```

**Gemini Pro** - Higher capability model for difficult documents

```bash
word-ocr document.docx -o ./output --engine gemini-pro
```
```

**Step 2: Update Library section with GeminiOCR example**

```markdown
# Add after the existing Library examples:

### Custom OCR Engine

```python
from word_ocr import process_document, GeminiOCR

# Use Gemini for better accuracy
engine = GeminiOCR()
result = process_document("report.docx", output_dir="./output", ocr_engine=engine)

# Use Gemini Pro for complex documents
engine = GeminiOCR(model="gemini-1.5-pro")
result = process_document("scanned.docx", output_dir="./output", ocr_engine=engine)
```
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Gemini OCR engine documentation"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Dependency setup | - |
| 2 | GeminiOCR API key validation | 3 |
| 3 | GeminiOCR model configuration | 2 |
| 4 | GeminiOCR text extraction | 2 |
| 5 | Package exports | - |
| 6 | CLI --engine option | 3 |
| 7 | README documentation | - |

**Total: 10 new tests**
**Expected final count: ~39 tests**
