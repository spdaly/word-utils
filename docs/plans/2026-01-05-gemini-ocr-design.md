# Gemini OCR Engine Design

Add a Google Gemini-based OCR engine for improved accuracy over Tesseract.

## Requirements

- **Primary goal**: Better OCR accuracy using Gemini's vision model
- **Future consideration**: Semantic understanding (not in this iteration)
- **API key**: Environment variable (`GEMINI_API_KEY` or `GOOGLE_API_KEY` fallback)
- **Model**: Configurable, defaults to `gemini-1.5-flash`
- **Error handling**: Fail fast (raise exceptions, let processor handle)
- **Dependency**: Required (add `google-generativeai` to main deps)

## Interface

### Library API

```python
from word_ocr import process_document, GeminiOCR

# Use Gemini instead of default Tesseract
engine = GeminiOCR()  # Uses gemini-1.5-flash
result = process_document("report.docx", output_dir="./output", ocr_engine=engine)

# Use Pro model for complex documents
engine = GeminiOCR(model="gemini-1.5-pro")
result = process_document("scanned.docx", output_dir="./output", ocr_engine=engine)
```

### CLI

```bash
# Use Gemini for all OCR
word-ocr document.docx -o ./output --engine gemini

# Use Gemini Pro
word-ocr document.docx -o ./output --engine gemini-pro
```

## Implementation

### GeminiOCR Class

```python
class GeminiOCR(OCREngine):
    """Google Gemini-based OCR implementation."""

    DEFAULT_MODEL = "gemini-1.5-flash"
    PROMPT = "Extract all text from this image exactly as it appears. Return only the extracted text, no commentary."

    def __init__(self, model: str = None):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def extract_text(self, image: Union[Path, Image.Image]) -> str:
        if isinstance(image, Path):
            image = Image.open(image)

        response = self.client.generate_content([self.PROMPT, image])
        return response.text
```

### Dependencies

```toml
# pyproject.toml
dependencies = [
    # ... existing deps
    "google-generativeai>=0.5.0",
]
```

### CLI Changes

```python
@click.option(
    "--engine",
    type=click.Choice(["tesseract", "gemini", "gemini-pro"]),
    default="tesseract",
    help="OCR engine to use"
)
def main(input_pattern: str, output: str, verbose: bool, dry_run: bool, engine: str):
    if engine == "tesseract":
        if not check_tesseract():
            # existing error handling
        ocr_engine = TesseractOCR()
    elif engine == "gemini":
        ocr_engine = GeminiOCR()
    elif engine == "gemini-pro":
        ocr_engine = GeminiOCR(model="gemini-1.5-pro")
```

## Error Handling

| Error | Behavior |
|-------|----------|
| No API key | `ValueError` on `GeminiOCR()` init with clear message |
| Invalid API key | Gemini SDK exception propagates |
| Rate limited | Gemini SDK exception propagates |
| Network error | Gemini SDK exception propagates |
| Empty image | Returns empty string |

All exceptions propagate up. The processor catches them and marks the image as `[OCR failed for this image]`.

## Testing

### Unit Tests

- `test_raises_without_api_key` - ValueError if no API key
- `test_accepts_gemini_api_key` - Uses GEMINI_API_KEY env var
- `test_falls_back_to_google_api_key` - Falls back to GOOGLE_API_KEY
- `test_uses_custom_model` - Respects model parameter
- `test_default_model_is_flash` - Defaults to gemini-1.5-flash

### Approach

- Mock `google.generativeai` to avoid real API calls
- API key validation tests run without mocking
- Optional `@pytest.mark.integration` test with real API (skipped in CI)

## Files Changed

- `src/word_ocr/ocr.py` - Add GeminiOCR class
- `src/word_ocr/__init__.py` - Export GeminiOCR
- `src/word_ocr/cli.py` - Add --engine option
- `tests/test_ocr.py` - Add GeminiOCR tests
- `tests/test_cli.py` - Add engine option tests
- `pyproject.toml` - Add google-generativeai dependency
- `README.md` - Document Gemini usage
