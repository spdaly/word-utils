# word-utils

A collection of utilities for working with Microsoft Word documents.

## Utilities

### word-ocr

Extract images from Word documents and OCR text into Markdown files.

**Use cases:**
- Convert scanned document archives to searchable text
- Extract screenshots and diagrams from reports
- Batch process documentation for text indexing

## Installation

**Requirements:**
- Python 3.10+
- Tesseract OCR

```bash
# Install Tesseract
brew install tesseract  # macOS
apt install tesseract-ocr  # Linux

# Install word-utils
uv pip install -e .
```

## Usage

### CLI

```bash
# Process a single file
word-ocr document.docx -o ./output

# Batch process multiple files
word-ocr "./docs/*.docx" -o ./output

# Show progress
word-ocr "./docs/*.docx" -o ./output -v

# Preview files without processing
word-ocr "./docs/*.docx" -o ./output --dry-run
```

**Exit codes:**
- `0` - All files processed successfully
- `1` - Some files failed
- `2` - All files failed or Tesseract not installed

### Library

```python
from word_ocr import process_document, process_batch

# Single document
result = process_document("report.docx", output_dir="./output")
print(f"Created: {result.markdown_path}")
print(f"Images: {result.image_count}")

# Batch processing
results = process_batch("./docs/*.docx", output_dir="./output")
for r in results:
    if r.success:
        print(f"OK: {r.markdown_path}")
    else:
        print(f"FAIL: {r.source_path}: {r.error}")
```

### Custom OCR Engine

```python
from word_ocr import process_document, OCREngine

class MyOCREngine(OCREngine):
    def extract_text(self, image):
        # Your OCR implementation
        return "extracted text"

result = process_document(
    "report.docx",
    output_dir="./output",
    ocr_engine=MyOCREngine()
)
```

## Output Format

For each Word document `report.docx`, creates:

```
output/
├── report.md
└── report_images/
    ├── image_001.png
    ├── image_002.png
    └── image_003.png
```

The Markdown file contains:

```markdown
# report.docx

## Image 1

![Image 1](report_images/image_001.png)

Extracted OCR text from the image...

---

## Image 2

![Image 2](report_images/image_002.png)

More extracted text...
```

## Development

```bash
# Clone and install
git clone <repo-url>
cd word-utils
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=word_ocr

# Current coverage: 96%
```

## Project Structure

```
word-utils/
├── src/
│   └── word_ocr/
│       ├── __init__.py      # Public API
│       ├── cli.py           # CLI interface
│       ├── extractor.py     # Image extraction
│       ├── ocr.py           # OCR abstraction
│       ├── processor.py     # Pipeline orchestration
│       └── renderer.py      # Markdown generation
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_cli.py
│   ├── test_extractor.py
│   ├── test_integration.py
│   ├── test_ocr.py
│   ├── test_processor.py
│   └── test_renderer.py
├── docs/
│   └── plans/               # Design documents
├── pyproject.toml
└── README.md
```

## License

MIT
