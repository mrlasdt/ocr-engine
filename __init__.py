# # Define package-level variables
# __version__ = '0.0'

# Import modules
from .src.ocr import OcrEngine
# from .src.word_formation import words_to_lines
from .src.word_formation import words_to_lines_tesseract as words_to_lines
from .src.utils import ImageReader, read_ocr_result_from_txt
from .src.dto import Word, Line, Page, Document, Box
# Expose package contents
__all__ = ["OcrEngine", "Word", "Line", "Page", "Document", "words_to_lines", "ImageReader", "read_ocr_result_from_txt", "Box"]

