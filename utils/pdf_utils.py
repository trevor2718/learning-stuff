"""
Utilities for PDF file handling and text extraction.
"""

import re
from pathlib import Path
from typing import List, Optional
import base64

from config import config

from PyPDF2 import PdfReader, PdfWriter

def get_pages(file_path : Path, page_numbers : List = [0]) -> Path:
    reader = PdfReader(file_path)
    writer = PdfWriter()

    for page_num in page_numbers:
        writer.add_page(reader.pages[page_num])
        
    output_path = config.ESSENTIALS_DIR / f"{file_path.name}"
    
    with open(output_path, "wb") as shortened_pdf:
        writer.write(shortened_pdf)
    
    return output_path


def natural_keys(s: str) -> List:
    """
    Sort key function for natural sorting of strings containing numbers.
    Args:
        s: String to parse
    
    Returns:
        List of alternating strings and integers for sorting
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]


def list_pdf_files(folder_path: str | Path) -> List[Path]:
    """
    Recursively list all PDF files in the given folder.
    
    Args:
        folder_path: Path to the folder to search
    
    Returns:
        List of Path objects for PDF files, sorted naturally
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all PDF files recursively
    pdf_files = list(folder_path.rglob("*.pdf"))
    
    # Sort using natural sorting
    pdf_files.sort(key=lambda p: natural_keys(str(p)))
    
    return pdf_files


def extract_pdf_text(
    file_path: str | Path,
    *,
    max_pages: Optional[int] = None,
    max_chars: int = 120_000,
) -> str:
    """
    Extract plain text from a PDF using PyPDF2.

    Args:
        file_path: Path to the PDF
        max_pages: If set, only read the first N pages
        max_chars: Truncate combined text beyond this length (API limits)
    """
    file_path = Path(file_path)
    reader = PdfReader(file_path)
    n = len(reader.pages)
    end = n if max_pages is None else min(n, max_pages)
    parts: List[str] = []
    for i in range(end):
        t = reader.pages[i].extract_text()
        if t:
            parts.append(t.strip())
    out = "\n\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n\n[... document truncated for API limits ...]"
    return out


def validate_pdf_file(file_path: str | Path) -> Path:
    """
    Validate that a file exists and is a PDF.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Path object if valid
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PDF
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")
    
    return file_path


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)

def read_pdf_as_base64(file_path: Path) -> str:
        """
        Read PDF file and convert to base64
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Base64 encoded string
        """
        # logger.debug(f"Reading PDF file: {file_path}")
        
        with open(file_path, "rb") as f:
            pdf_data = f.read()
        
        return base64.standard_b64encode(pdf_data).decode("utf-8")
    