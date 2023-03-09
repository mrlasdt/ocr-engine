# # Define package-level variables
# __version__ = '0.0'

# Import modules
from .src.ocr import OcrEngine
from .src.word_formation import Word, Line, words_to_lines
from .src.utils import ImageReader
from .src.dto import Word, Line, Page, Document
# Expose package contents
__all__ = ["OcrEngine", "Word", "Line", "Page", "Document", "words_to_lines", "ImageReader"]
