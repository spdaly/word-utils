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
