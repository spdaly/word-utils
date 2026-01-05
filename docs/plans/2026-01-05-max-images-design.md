# Max Images per Document Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--max-images N` CLI option to process only the first N images from each Word document.

**Architecture:** Add optional parameter that flows from CLI through processor. Slice extracted images before OCR loop.

**Tech Stack:** Click for CLI option, Python slicing for limit.

---

## Changes

### 1. CLI (`cli.py`)

Add `--max-images` option:
- Type: integer
- Optional (no default = process all)
- Pass to `process_document()`

### 2. Processor (`processor.py`)

Add `max_images` parameter to `process_document()`:
- Optional[int], default None
- After extraction, slice: `extracted[:max_images]` if set
- Track total extracted count for reporting

### 3. Verbose Output

When images are skipped, show:
- "(3 of 12 images, limited)" instead of "(3 images)"

## Example Usage

```bash
# Limit to 3 images per document
word-ocr docs/*.docx -o output/ --max-images 3

# No limit (current behavior)
word-ocr docs/*.docx -o output/
```

## Testing

- Test with `--max-images 2` on doc with 5 images -> processes 2
- Test with `--max-images 10` on doc with 3 images -> processes 3
- Test without option -> processes all (no regression)
