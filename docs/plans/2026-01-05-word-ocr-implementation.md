# Word OCR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python utility that extracts images from Word documents, OCRs them with Tesseract, and outputs Markdown files with inline image references.

**Architecture:** Library-first design with three core components (Extractor, OCR, Renderer) and a thin CLI wrapper. Each component is independently testable. Batch processing iterates through documents, processing each through the full pipeline.

**Tech Stack:** Python 3.10+, python-docx, pytesseract, Pillow, click, pytest

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/word_ocr/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "word-ocr"
version = "0.1.0"
description = "Extract images from Word documents and OCR to Markdown"
requires-python = ">=3.10"
dependencies = [
    "python-docx>=1.1.0",
    "pytesseract>=0.3.10",
    "Pillow>=10.0.0",
    "click>=8.1.0",
]

[project.scripts]
word-ocr = "word_ocr.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.0.0"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2: Create package structure**

```bash
mkdir -p src/word_ocr tests
touch src/word_ocr/__init__.py tests/__init__.py
```

**Step 3: Install dependencies**

Run: `uv pip install -e ".[dev]"`
Expected: Successfully installed all packages

**Step 4: Verify setup**

Run: `uv run python -c "import word_ocr; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "chore: initial project setup with dependencies"
```

---

## Task 2: OCR Engine Abstraction

**Files:**
- Create: `src/word_ocr/ocr.py`
- Create: `tests/test_ocr.py`

**Step 1: Write the failing test for OCR interface**

```python
# tests/test_ocr.py
# ABOUTME: Tests for OCR engine abstraction
# ABOUTME: Validates Tesseract wrapper and interface contract

import pytest
from pathlib import Path
from PIL import Image
import io

from word_ocr.ocr import OCREngine, TesseractOCR


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/word_ocr/ocr.py
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/word_ocr/ocr.py tests/test_ocr.py
git commit -m "feat: add OCR engine abstraction with Tesseract implementation"
```

---

## Task 3: Word Document Image Extractor

**Files:**
- Create: `src/word_ocr/extractor.py`
- Create: `tests/test_extractor.py`
- Create: `tests/fixtures/` (directory for test documents)

**Step 1: Create test fixture - Word document with images**

```python
# tests/conftest.py
# ABOUTME: Pytest fixtures for word-ocr tests
# ABOUTME: Provides sample Word documents with embedded images

import pytest
from pathlib import Path
from docx import Document
from docx.shared import Inches
from PIL import Image
import io


@pytest.fixture
def sample_docx_with_images(tmp_path):
    """Create a Word document with embedded images."""
    doc = Document()
    doc.add_heading('Test Document', 0)

    # Create and add first image
    img1 = Image.new('RGB', (100, 100), color='red')
    img1_bytes = io.BytesIO()
    img1.save(img1_bytes, format='PNG')
    img1_bytes.seek(0)

    doc.add_paragraph('First paragraph with image:')
    doc.add_picture(img1_bytes, width=Inches(1.0))

    # Create and add second image
    img2 = Image.new('RGB', (100, 100), color='blue')
    img2_bytes = io.BytesIO()
    img2.save(img2_bytes, format='PNG')
    img2_bytes.seek(0)

    doc.add_paragraph('Second paragraph with image:')
    doc.add_picture(img2_bytes, width=Inches(1.0))

    # Save document
    doc_path = tmp_path / "test_with_images.docx"
    doc.save(doc_path)

    return doc_path


@pytest.fixture
def sample_docx_no_images(tmp_path):
    """Create a Word document without images."""
    doc = Document()
    doc.add_heading('Test Document', 0)
    doc.add_paragraph('Just text, no images.')

    doc_path = tmp_path / "test_no_images.docx"
    doc.save(doc_path)

    return doc_path


@pytest.fixture
def empty_docx(tmp_path):
    """Create an empty Word document."""
    doc = Document()

    doc_path = tmp_path / "empty.docx"
    doc.save(doc_path)

    return doc_path
```

**Step 2: Write the failing test for extractor**

```python
# tests/test_extractor.py
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
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: FAIL with import error

**Step 4: Write minimal implementation**

```python
# src/word_ocr/extractor.py
# ABOUTME: Extracts embedded images from Word (.docx) documents
# ABOUTME: Returns list of ExtractedImage with PIL Image and metadata

from dataclasses import dataclass
from pathlib import Path
from typing import List
import io

from docx import Document
from PIL import Image


@dataclass
class ExtractedImage:
    """An image extracted from a Word document."""

    image: Image.Image
    index: int
    content_type: str = "image/png"


class ImageExtractor:
    """Extracts images from Word documents."""

    def extract(self, docx_path: Path) -> List[ExtractedImage]:
        """Extract all images from a Word document.

        Args:
            docx_path: Path to .docx file

        Returns:
            List of ExtractedImage objects in document order

        Raises:
            FileNotFoundError: If document doesn't exist
        """
        docx_path = Path(docx_path)

        if not docx_path.exists():
            raise FileNotFoundError(f"Document not found: {docx_path}")

        doc = Document(docx_path)
        images = []
        index = 1

        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_data = rel.target_part.blob
                pil_image = Image.open(io.BytesIO(image_data))
                content_type = rel.target_part.content_type

                images.append(ExtractedImage(
                    image=pil_image,
                    index=index,
                    content_type=content_type
                ))
                index += 1

        return images
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/word_ocr/extractor.py tests/test_extractor.py tests/conftest.py
git commit -m "feat: add Word document image extractor"
```

---

## Task 4: Markdown Renderer

**Files:**
- Create: `src/word_ocr/renderer.py`
- Create: `tests/test_renderer.py`

**Step 1: Write the failing test for renderer**

```python
# tests/test_renderer.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/word_ocr/renderer.py
# ABOUTME: Generates Markdown output from processed images
# ABOUTME: Creates .md file with inline image references and OCR text

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image


@dataclass
class ProcessedImage:
    """An image with its OCR result."""

    image: Image.Image
    index: int
    ocr_text: Optional[str]


@dataclass
class RenderResult:
    """Result of rendering a document."""

    markdown_path: Path
    images_dir: Path


class MarkdownRenderer:
    """Renders processed images to Markdown format."""

    def render(
        self,
        images: List[ProcessedImage],
        source_name: str,
        output_dir: Path
    ) -> RenderResult:
        """Render images and OCR text to Markdown.

        Args:
            images: List of processed images with OCR text
            source_name: Original document filename
            output_dir: Directory to write output

        Returns:
            RenderResult with paths to created files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive names from source
        base_name = Path(source_name).stem
        md_path = output_dir / f"{base_name}.md"
        images_dir = output_dir / f"{base_name}_images"

        # Build markdown content
        lines = [f"# {source_name}", ""]

        if not images:
            lines.append("No images found in this document.")
        else:
            images_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                # Save image
                img_filename = f"image_{img.index:03d}.png"
                img_path = images_dir / img_filename
                img.image.save(img_path, "PNG")

                # Add to markdown
                lines.append(f"## Image {img.index}")
                lines.append("")
                lines.append(f"![Image {img.index}]({base_name}_images/{img_filename})")
                lines.append("")

                if img.ocr_text is None:
                    lines.append("[OCR failed for this image]")
                else:
                    lines.append(img.ocr_text.strip())

                lines.append("")
                lines.append("---")
                lines.append("")

        # Write markdown file
        md_path.write_text("\n".join(lines))

        return RenderResult(
            markdown_path=md_path,
            images_dir=images_dir if images else output_dir
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/word_ocr/renderer.py tests/test_renderer.py
git commit -m "feat: add Markdown renderer for OCR output"
```

---

## Task 5: Document Processor (Integration)

**Files:**
- Create: `src/word_ocr/processor.py`
- Create: `tests/test_processor.py`
- Modify: `src/word_ocr/__init__.py`

**Step 1: Write the failing integration test**

```python
# tests/test_processor.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_processor.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/word_ocr/processor.py
# ABOUTME: Main document processing pipeline
# ABOUTME: Orchestrates extraction, OCR, and rendering for single/batch processing

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from glob import glob

from .extractor import ImageExtractor
from .ocr import OCREngine, TesseractOCR
from .renderer import MarkdownRenderer, ProcessedImage


@dataclass
class ProcessResult:
    """Result of processing a single document."""

    source_path: Path
    success: bool
    markdown_path: Optional[Path] = None
    images_dir: Optional[Path] = None
    image_count: int = 0
    error: Optional[str] = None


def process_document(
    input_path: Path,
    output_dir: Path,
    ocr_engine: Optional[OCREngine] = None
) -> ProcessResult:
    """Process a single Word document.

    Args:
        input_path: Path to .docx file
        output_dir: Directory for output files
        ocr_engine: OCR engine to use (defaults to Tesseract)

    Returns:
        ProcessResult with status and output paths
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if ocr_engine is None:
        ocr_engine = TesseractOCR()

    try:
        # Extract images
        extractor = ImageExtractor()
        extracted = extractor.extract(input_path)

        # OCR each image
        processed = []
        for img in extracted:
            try:
                text = ocr_engine.extract_text(img.image)
            except Exception:
                text = None

            processed.append(ProcessedImage(
                image=img.image,
                index=img.index,
                ocr_text=text
            ))

        # Render output
        renderer = MarkdownRenderer()
        result = renderer.render(
            images=processed,
            source_name=input_path.name,
            output_dir=output_dir
        )

        return ProcessResult(
            source_path=input_path,
            success=True,
            markdown_path=result.markdown_path,
            images_dir=result.images_dir if processed else None,
            image_count=len(processed)
        )

    except FileNotFoundError as e:
        return ProcessResult(
            source_path=input_path,
            success=False,
            error=f"File not found: {input_path}"
        )
    except Exception as e:
        return ProcessResult(
            source_path=input_path,
            success=False,
            error=str(e)
        )


def process_batch(
    input_pattern: str,
    output_dir: Path,
    ocr_engine: Optional[OCREngine] = None
) -> List[ProcessResult]:
    """Process multiple Word documents matching a pattern.

    Args:
        input_pattern: Glob pattern for input files
        output_dir: Directory for output files
        ocr_engine: OCR engine to use (defaults to Tesseract)

    Returns:
        List of ProcessResult, one per document
    """
    files = glob(input_pattern)
    results = []

    for file_path in sorted(files):
        result = process_document(
            input_path=Path(file_path),
            output_dir=output_dir,
            ocr_engine=ocr_engine
        )
        results.append(result)

    return results
```

**Step 4: Update `__init__.py` with public API**

```python
# src/word_ocr/__init__.py
# ABOUTME: Word OCR public API exports
# ABOUTME: Exposes process_document and process_batch functions

from .processor import process_document, process_batch, ProcessResult
from .ocr import OCREngine, TesseractOCR
from .extractor import ImageExtractor, ExtractedImage
from .renderer import MarkdownRenderer, ProcessedImage, RenderResult

__all__ = [
    "process_document",
    "process_batch",
    "ProcessResult",
    "OCREngine",
    "TesseractOCR",
    "ImageExtractor",
    "ExtractedImage",
    "MarkdownRenderer",
    "ProcessedImage",
    "RenderResult",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_processor.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add src/word_ocr/processor.py src/word_ocr/__init__.py tests/test_processor.py
git commit -m "feat: add document processor with batch support"
```

---

## Task 6: CLI Interface

**Files:**
- Create: `src/word_ocr/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test for CLI**

```python
# tests/test_cli.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/word_ocr/cli.py
# ABOUTME: CLI entry point for word-ocr
# ABOUTME: Provides command-line interface for batch document processing

import shutil
import sys
from glob import glob
from pathlib import Path

import click

from .processor import process_document, process_batch


def check_tesseract() -> bool:
    """Check if Tesseract is installed."""
    return shutil.which("tesseract") is not None


@click.command()
@click.argument("input_pattern")
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for Markdown files"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show progress information"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files that would be processed without processing"
)
def main(input_pattern: str, output: str, verbose: bool, dry_run: bool):
    """Extract images from Word documents and OCR to Markdown.

    INPUT_PATTERN can be a single file or glob pattern (e.g., "docs/*.docx")
    """
    # Check Tesseract is available
    if not check_tesseract():
        click.echo("Error: Tesseract is not installed.", err=True)
        click.echo("Install with: brew install tesseract (macOS)", err=True)
        click.echo("             apt install tesseract-ocr (Linux)", err=True)
        sys.exit(2)

    # Find matching files
    files = glob(input_pattern)
    if not files:
        # Maybe it's a single file path
        if Path(input_pattern).exists():
            files = [input_pattern]
        else:
            click.echo(f"No files match pattern: {input_pattern}", err=True)
            sys.exit(2)

    # Dry run - just list files
    if dry_run:
        click.echo("Files that would be processed:")
        for f in sorted(files):
            click.echo(f"  {Path(f).name}")
        click.echo(f"\nTotal: {len(files)} file(s)")
        return

    # Process files
    output_dir = Path(output)
    total = len(files)
    successes = 0
    failures = 0

    for i, file_path in enumerate(sorted(files), 1):
        file_name = Path(file_path).name

        if verbose:
            click.echo(f"Processing {i}/{total}: {file_name}")

        result = process_document(
            input_path=file_path,
            output_dir=output_dir
        )

        if result.success:
            successes += 1
            if verbose:
                click.echo(f"  -> {result.markdown_path.name} ({result.image_count} images)")
        else:
            failures += 1
            click.echo(f"Error processing {file_name}: {result.error}", err=True)

    # Summary
    click.echo(f"\nProcessed {total} file(s): {successes} succeeded, {failures} failed")

    # Exit code
    if failures == total:
        sys.exit(2)
    elif failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/word_ocr/cli.py tests/test_cli.py
git commit -m "feat: add CLI interface with batch support"
```

---

## Task 7: Final Integration Test & Documentation

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

**Step 1: Write end-to-end integration test**

```python
# tests/test_integration.py
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
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Create README.md**

```markdown
# word-ocr

Extract images from Word documents and OCR text into Markdown files.

## Installation

```bash
# Install Tesseract (required)
brew install tesseract  # macOS
apt install tesseract-ocr  # Linux

# Install word-ocr
uv pip install -e .
```

## Usage

### CLI

```bash
# Single file
word-ocr document.docx -o ./output

# Batch processing
word-ocr "./docs/*.docx" -o ./output

# With progress output
word-ocr "./docs/*.docx" -o ./output -v

# Dry run (list files without processing)
word-ocr "./docs/*.docx" --dry-run
```

### Library

```python
from word_ocr import process_document, process_batch

# Single document
result = process_document("report.docx", output_dir="./output")
print(f"Created: {result.markdown_path}")

# Batch processing
results = process_batch("./docs/*.docx", output_dir="./output")
for r in results:
    if r.success:
        print(f"OK: {r.markdown_path}")
    else:
        print(f"FAIL: {r.source_path}: {r.error}")
```

## Output Format

For each Word document, creates:
- `<name>.md` - Markdown with image references and OCR text
- `<name>_images/` - Extracted images

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=word_ocr
```
```

**Step 4: Run all tests**

Run: `uv run pytest --cov=word_ocr`
Expected: All tests pass with >90% coverage

**Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "docs: add README and integration tests"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project setup | - |
| 2 | OCR engine | 4 |
| 3 | Image extractor | 6 |
| 4 | Markdown renderer | 4 |
| 5 | Document processor | 5 |
| 6 | CLI | 7 |
| 7 | Integration | 1 |

**Total: 27 tests**
