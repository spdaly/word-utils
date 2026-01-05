# Word OCR Utility Design

Extract images from Word documents and OCR text into Markdown files.

## Requirements

- **Batch processing** - Process many Word documents in one run
- **Mixed content** - Handle screenshots, scanned documents, diagrams, charts
- **Output format** - One Markdown file per Word document, images preserved in adjacent folder
- **OCR engine** - Tesseract (local, no API costs), pluggable for future alternatives
- **Language** - Python with library-first design and CLI wrapper

## Project Structure

```
word-utils/
├── src/
│   └── word_ocr/
│       ├── __init__.py        # Public API exports
│       ├── cli.py             # CLI entry point
│       ├── extractor.py       # Word document image extraction
│       ├── ocr.py             # OCR engine abstraction
│       └── renderer.py        # Markdown output generation
├── tests/
│   ├── fixtures/              # Sample .docx files for testing
│   ├── test_extractor.py
│   ├── test_ocr.py
│   └── test_renderer.py
├── pyproject.toml
└── README.md
```

## Components

### Extractor

Opens `.docx` files using `python-docx`, iterates through embedded images, extracts them with metadata (position in document, original filename if available).

### OCR Engine

Abstract interface with Tesseract as default implementation. Takes an image, returns extracted text. Designed for swapping in different engines without changing calling code.

### Renderer

Takes extracted images and OCR results, generates a single Markdown file per document. Images saved to `<docname>_images/` folder, referenced inline in the Markdown.

## Data Flow

```
document.docx
    ↓
┌─────────────────────┐
│  Extractor          │  → extracts images in document order
└─────────────────────┘
    ↓
    [Image 1, Image 2, Image 3, ...]
    ↓
┌─────────────────────┐
│  OCR Engine         │  → processes each image
└─────────────────────┘
    ↓
    [(Image 1, "text..."), (Image 2, "text..."), ...]
    ↓
┌─────────────────────┐
│  Renderer           │  → generates output
└─────────────────────┘
    ↓
output/
├── document.md
└── document_images/
    ├── image_001.png
    ├── image_002.png
    └── image_003.png
```

## Output Format

```markdown
# document.docx

## Image 1

![Image 1](document_images/image_001.png)

extracted text from image 1 goes here...

---

## Image 2

![Image 2](document_images/image_002.png)

extracted text from image 2 goes here...
```

## Library API

```python
from word_ocr import process_document, process_batch

# Single document
result = process_document(
    input_path="report.docx",
    output_dir="./output",
    ocr_engine=None  # defaults to Tesseract
)
# Returns: ProcessResult with paths to .md and images

# Batch processing
results = process_batch(
    input_pattern="./docs/*.docx",
    output_dir="./output",
    ocr_engine=None
)
# Returns: list of ProcessResult, one per document
```

## CLI Interface

```bash
# Single file
word-ocr report.docx -o ./output

# Batch with glob pattern
word-ocr "./docs/*.docx" -o ./output

# With verbose output (shows progress)
word-ocr "./docs/*.docx" -o ./output -v

# Dry run (list files that would be processed)
word-ocr "./docs/*.docx" --dry-run
```

**Exit codes:**
- `0` - Success
- `1` - Partial failure (some documents failed, others succeeded)
- `2` - Complete failure (no documents processed)

## Error Handling

### Document-level errors (non-fatal in batch mode)

- Corrupted/unreadable `.docx` file → Log error, skip to next document
- No images in document → Generate empty `.md` with note "No images found"
- Permission denied on input → Log error, skip to next document

### Image-level errors (non-fatal)

- OCR fails on specific image → Include image in output, placeholder text: `[OCR failed for this image]`
- Corrupted/unreadable image data → Log warning, skip image, continue with others

### Output errors (fatal)

- Cannot create output directory → Fail immediately with clear error message
- Disk full → Fail immediately

### Edge cases

- Duplicate image filenames → Append sequential numbers: `image_001.png`, `image_002.png`
- Very large images → Process as-is (Tesseract handles them)
- Non-image embedded objects (charts, SmartArt) → Skip with warning logged
- Empty Word document → Generate `.md` with note "Empty document"

### Progress reporting

- Batch mode prints progress: `Processing document 3/15: report.docx`
- Verbose mode (`-v`) shows per-image progress
- All errors/warnings go to stderr, normal output to stdout

## Testing Strategy

### Unit tests

- `test_extractor.py` - Test image extraction with sample `.docx` files
  - Document with multiple images
  - Document with no images
  - Document with various image formats (PNG, JPEG, etc.)

- `test_ocr.py` - Test OCR engine abstraction
  - Mock Tesseract for fast tests
  - Integration test with real Tesseract on known images

- `test_renderer.py` - Test Markdown generation
  - Correct image references
  - Proper escaping of special characters in OCR text
  - Output file/folder naming

### Integration tests

- End-to-end processing of sample documents
- Batch processing with mix of valid/invalid files
- Verify output structure matches specification

### Test fixtures

- `tests/fixtures/` containing sample `.docx` files with known content
- Sample images with known OCR output for validation

### Commands

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=word_ocr
```

Target: >90% coverage on core logic.

## Dependencies

```toml
[project]
name = "word-ocr"
version = "0.1.0"
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
```

### System requirement

Tesseract must be installed separately:
- macOS: `brew install tesseract`
- Linux: `apt install tesseract-ocr`

The CLI checks for Tesseract on startup and provides a helpful error if missing.

## Installation

```bash
# From source
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```
