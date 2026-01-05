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
