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
