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
