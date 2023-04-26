# %%
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add Fiintrade/ to path
# %%
bug_case = '/home/sds/namnt/FWD/inter_results/bug_case.pkl'
with open(bug_case, 'rb') as f:
    all_documents = pickle.load(f)
page = all_documents[0]
img = page['image']
boxes = page['boxes']
texts = page['contents']
plt.figure(figsize=(20, 20))
plt.imshow(img)
# %%
from src.dto import Word, Box
lwords = list()
for i in range(len(texts)):
    text = texts[i] 
    bbox = Box(*boxes[i]) if isinstance(boxes[i], list) else boxes[i]
    lwords.append(Word(image=img, text=text, conf_cls=-1, bndbox=bbox, conf_detect=bbox.conf))
# %%

from src.word_formation import words_to_lines_tesseract
llines = words_to_lines_tesseract(lwords, 0.6, 20, 0.5)[0]
llines

# %%
