# ABOUTME: CLI entry point for word-ocr
# ABOUTME: Provides command-line interface for batch document processing

import shutil
import sys
from glob import glob
from pathlib import Path

import click
from dotenv import load_dotenv

from .processor import process_document
from .ocr import TesseractOCR, GeminiOCR

# Load environment variables from .env file
load_dotenv()


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
@click.option(
    "--engine",
    type=click.Choice(["tesseract", "gemini", "gemini-pro"]),
    default="tesseract",
    help="OCR engine to use (default: tesseract)"
)
@click.option(
    "--max-images",
    type=int,
    default=None,
    help="Maximum images to process per document"
)
def main(input_pattern: str, output: str, verbose: bool, dry_run: bool, engine: str, max_images: int):
    """Extract images from Word documents and OCR to Markdown.

    INPUT_PATTERN can be a single file or glob pattern (e.g., "docs/*.docx")
    """
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

    # Select OCR engine
    if engine == "tesseract":
        if not check_tesseract():
            click.echo("Error: Tesseract is not installed.", err=True)
            click.echo("Install with: brew install tesseract (macOS)", err=True)
            click.echo("             apt install tesseract-ocr (Linux)", err=True)
            sys.exit(2)
        ocr_engine = TesseractOCR()
    elif engine == "gemini":
        try:
            ocr_engine = GeminiOCR()
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"Unexpected error initializing Gemini OCR engine: {e}", err=True)
            sys.exit(2)
    elif engine == "gemini-pro":
        try:
            ocr_engine = GeminiOCR(model="gemini-1.5-pro")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"Unexpected error initializing Gemini OCR engine (gemini-pro): {e}", err=True)
            sys.exit(2)

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
            output_dir=output_dir,
            ocr_engine=ocr_engine,
            max_images=max_images
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
