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
