#%%
from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add Fiintrade/ to path
#%%

from src.ocr import OcrEngine
# img_path = "data/PH/Sea7/Sea_7_1.jpg"
img_path = "/mnt/hdd2T/AICR/Projects/2023/FWD/Email/email_DDMMYY_HHSS_12042023_153953058.jpg"
engine = OcrEngine(device="cuda:1")
# https://stackoverflow.com/questions/66435480/overload-following-optional-argument
page = engine(img_path)  # type: ignore
#%%
print(page.llines)
#%%
from src.utils import visualize_bbox_and_label
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(img_path)
img = engine.preprocess(img)
bboxes = []
texts = []
for line in page.llines:
    for wg in line.list_word_groups:
        bboxes.append(wg.bbox[:])
        texts.append(wg.text)
        # for w in wg.list_words:
        #     bboxes.append(w.bbox[:])
        #     texts.append(w.text)   
vis = visualize_bbox_and_label(img, bboxes, texts,is_vnese=True,)
plt.figure(figsize=(20,20))
plt.imshow(vis)
      

# %%
plt.figure(figsize=(20,20))
plt.imshow(page.visualize_bbox_and_label())


# %%
