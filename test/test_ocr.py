# from pathlib import Path
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[1].as_posix())  # add Fiintrade/ to path


from externals.ocr_sdsv import OcrEngine
img_path = "data/PH/Sea7/Sea_7_1.jpg"
engine = OcrEngine()
# https://stackoverflow.com/questions/66435480/overload-following-optional-argument
page = engine(img_path)  # type: ignore
print(page.__llines)
