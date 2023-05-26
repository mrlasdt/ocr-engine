# %%
from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add Fiintrade/ to path
# %%

from src.ocr import OcrEngine
# img_path = "data/PH/Sea7/Sea_7_1.jpg"
# img_path = "/mnt/hdd2T/AICR/Projects/2023/FWD/Email/email_DDMMYY_HHSS_12042023_153953024.jpg"
# img_path = "/mnt/ssd1T/hungbnt/ocr_engine/results/00206BF7E1DF230403162311/00206BF7E1DF230403162311-1.jpg"
# img_path = "/mnt/hdd2T/AICR/Projects/2023/FWD/─Р├г c├│ Form ID (╞░u ti├кn cao h╞бn)/OCR040_TH╞п B├БO Chс║еm dс╗йt hiс╗Зu lс╗▒c Hс╗гp ─Сс╗Уng bс║гo hiс╗Гm_POS-19.pdf"
img_path = "/mnt/hdd2T/AICR/Datasets/So_ho_khau/hinh-anh-tam-tru-la-gi-so-5.jpg"


engine = OcrEngine(device="cuda:1")
# https://stackoverflow.com/questions/66435480/overload-following-optional-argument
page = engine(img_path)  # type: ignore
# %%
print(*page.word_groups, sep="\n")

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
plt.imshow(page.visualize_bbox_and_label(is_vnese=True))
# %%
from src.utils import visualize_bbox_and_label
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(img_path)
# img = engine.preprocess(img)
bboxes = []
texts = []
for wg in page.word_groups:
    bboxes.append(wg.bbox)
    texts.append(wg._text)
    # for wg in line.list_word_groups:
    # bboxes.append(wg.bbox)
    # texts.append(wg.text)
    # for w in wg.list_words:
    #     bboxes.append(w.bbox[:])
    #     texts.append(w.text)
vis = visualize_bbox_and_label(img, bboxes, texts, is_vnese=True,)
plt.figure(figsize=(20, 20))
plt.imshow(vis)


# %%
plt.figure(figsize=(20, 20))
plt.imshow(page.visualize_bbox_and_label())

# %%
page.word_groups
# %%
