import sys
import os

# # Add the project root (one level up from "tests") to Python’s import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyPDF2 import PdfReader
from pathlib import Path
from typing import List
from PyPDF2 import PdfWriter

from config import config

def get_pages(file_path : Path, page_numbers : List = [0]) -> bool:
    reader = PdfReader(file_path)
    writer = PdfWriter()
    try:
        for page_num in page_numbers:
            writer.add_page(reader.pages[page_num])
            
        output_path = config.ESSENTIALS_DIR / f"{file_path.name}"
        
        with open(output_path, "wb") as shortened_pdf:
            writer.write(shortened_pdf)
        
        return True
    
    except Exception as e:
        print(f"Error creating shortened PDF: {str(e)}")
        return False

test_paper = config.TRAINING_DIR / "pdf_1.pdf"

print(get_pages(test_paper))