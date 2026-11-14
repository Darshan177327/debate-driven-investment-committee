
import json
import csv
import io
import re
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any

import pymupdf as fitz
import pandas as pd

from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text


# =========================================================
# SUPPORTED FORMATS
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".csv",
    ".pptx",
    ".txt",
    ".md",
    ".rtf",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp"
}


# =========================================================
# COMMON TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# OCR ENGINE
# =========================================================

def extract_image_text(image_path: Path) -> str:

    try:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)

        extracted = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return clean_text(extracted)

    except Exception as error:

        print(f"[OCR WARNING] {error}")

        return ""


# =========================================================
# PDF PROCESSOR
# =========================================================

def extract_pdf(file_path: Path) -> list:

    pages = []

    document = fitz.open(file_path)

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        # Attempt OCR for scanned pages with little/no text.
        if len(text.strip()) < 30:

            try:

                import pytesseract
                from PIL import Image

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )

                ocr_text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )

                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text

            except Exception:
                pass

        pages.append({
            "page": page_number,
            "text": clean_text(text)
        })

    document.close()

    return pages


# =========================================================
# WORD PROCESSOR
# =========================================================

def extract_docx(file_path: Path) -> list:

    document = Document(file_path)

    content = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            content.append(paragraph.text)

    for table_number, table in enumerate(document.tables, start=1):

        content.append(f"\nTABLE {table_number}")

        for row in table.rows:

            values = [
                cell.text.replace("\n", " ").strip()
                for cell in row.cells
            ]

            content.append(" | ".join(values))

    return [{
        "page": None,
        "text": clean_text("\n".join(content))
    }]


# =========================================================
# EXCEL PROCESSOR
# =========================================================

def extract_excel(file_path: Path) -> list:

    pages = []

    workbook = pd.ExcelFile(file_path)

    for sheet_name in workbook.sheet_names:

        dataframe = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            dtype=str
        )

        dataframe = dataframe.fillna("")

        text = dataframe.to_string(
            index=False
        )

        pages.append({
            "page": sheet_name,
            "text": clean_text(
                f"WORKSHEET: {sheet_name}\n{text}"
            )
        })

    return pages


# =========================================================
# CSV PROCESSOR
# =========================================================

def extract_csv(file_path: Path) -> list:

    dataframe = pd.read_csv(
        file_path,
        dtype=str,
        encoding="utf-8-sig"
    )

    dataframe = dataframe.fillna("")

    return [{
        "page": "CSV",
        "text": clean_text(
            dataframe.to_string(index=False)
        )
    }]


# =========================================================
# POWERPOINT PROCESSOR
# =========================================================

def extract_pptx(file_path: Path) -> list:

    presentation = Presentation(file_path)

    pages = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        content = []

        for shape in slide.shapes:

            if hasattr(shape, "text") and shape.text.strip():

                content.append(shape.text)

            if shape.has_table:

                for row in shape.table.rows:

                    values = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    content.append(" | ".join(values))

        pages.append({
            "page": slide_number,
            "text": clean_text("\n".join(content))
        })

    return pages


# =========================================================
# TEXT PROCESSOR
# =========================================================

def extract_plain_text(file_path: Path) -> list:

    text = file_path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    )

    return [{
        "page": None,
        "text": clean_text(text)
    }]


# =========================================================
# RTF PROCESSOR
# =========================================================

def extract_rtf(file_path: Path) -> list:

    raw_text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    text = rtf_to_text(raw_text)

    return [{
        "page": None,
        "text": clean_text(text)
    }]


# =========================================================
# HTML PROCESSOR
# =========================================================

def extract_html(file_path: Path) -> list:

    html = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "noscript"
    ]):

        element.decompose()

    text = soup.get_text(
        separator="\n"
    )

    return [{
        "page": None,
        "text": clean_text(text)
    }]


# =========================================================
# JSON PROCESSOR
# =========================================================

def extract_json(file_path: Path) -> list:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    text = json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )

    return [{
        "page": None,
        "text": clean_text(text)
    }]


# =========================================================
# XML PROCESSOR
# =========================================================

def extract_xml(file_path: Path) -> list:

    tree = ET.parse(file_path)

    root = tree.getroot()

    text = ET.tostring(
        root,
        encoding="unicode"
    )

    return [{
        "page": None,
        "text": clean_text(text)
    }]


# =========================================================
# IMAGE PROCESSOR
# =========================================================

def extract_image(file_path: Path) -> list:

    text = extract_image_text(file_path)

    return [{
        "page": "Image",
        "text": text
    }]


# =========================================================
# UNIVERSAL DOCUMENT PROCESSOR
# =========================================================

def extract_document(file_path: str) -> dict[str, Any]:

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():

        raise ValueError(
            "The supplied path is not a file."
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    print("\n" + "=" * 55)

    print("[DOCUMENT PROCESSOR] Starting extraction")

    print("[FILE]", path.name)

    print("[FORMAT]", extension)

    # Select appropriate processor.
    processors = {

        ".pdf": extract_pdf,

        ".docx": extract_docx,

        ".xlsx": extract_excel,
        ".xls": extract_excel,

        ".csv": extract_csv,

        ".pptx": extract_pptx,

        ".txt": extract_plain_text,
        ".md": extract_plain_text,

        ".rtf": extract_rtf,

        ".html": extract_html,
        ".htm": extract_html,

        ".json": extract_json,

        ".xml": extract_xml,

        ".png": extract_image,
        ".jpg": extract_image,
        ".jpeg": extract_image,
        ".tiff": extract_image,
        ".tif": extract_image,
        ".bmp": extract_image
    }

    processor = processors[extension]

    pages = processor(path)

    full_text = "\n\n".join(
        str(page["text"])
        for page in pages
        if page["text"]
    )

    full_text = clean_text(full_text)

    if not full_text:

        raise ValueError(
            "No readable text found. "
            "The document may be encrypted, empty, "
            "or require additional OCR processing."
        )

    result = {

        "filename": path.name,

        "file_path": str(path.resolve()),

        "file_type": extension,

        "total_pages": len(pages),

        "character_count": len(full_text),

        "pages": pages,

        "full_text": full_text
    }

    print("[SUCCESS] Extraction completed")

    print("[PAGES/SHEETS/SLIDES]", len(pages))

    print("[CHARACTERS]", len(full_text))

    print("=" * 55)

    return result


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python -m tools.universal_document_processor "
            "<file_path>"
        )

    else:

        result = extract_document(sys.argv[1])

        print("\nEXTRACTED TEXT PREVIEW\n")

        print(
            result["full_text"][:3000]
        )
