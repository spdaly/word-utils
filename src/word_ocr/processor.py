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
    total_extracted: int = 0
    error: Optional[str] = None


def process_document(
    input_path: Path,
    output_dir: Path,
    ocr_engine: Optional[OCREngine] = None,
    max_images: Optional[int] = None
) -> ProcessResult:
    """Process a single Word document.

    Args:
        input_path: Path to .docx file
        output_dir: Directory for output files
        ocr_engine: OCR engine to use (defaults to Tesseract)
        max_images: Maximum number of images to process (None for all)

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
        total_extracted = len(extracted)

        # Apply limit if set
        if max_images is not None:
            extracted = extracted[:max_images]

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
            image_count=len(processed),
            total_extracted=total_extracted
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
