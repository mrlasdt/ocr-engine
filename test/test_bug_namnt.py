
# %%
import sys
sys.path.append("/home/sds/namnt/FWD")
import os
os.chdir("/home/sds/namnt/FWD")
# %%

import cv2
import glob
import os
from time import time
from datetime import datetime

from debug_predictions import vis_debugging
from src.ocr_master import Extractor


def test_auto_flow(images):
    result, inter_result = extractor(images, None)
    return result, inter_result


def test_manual_clf_flow(images, doc_groups):
    result, inter_result = extractor(images, doc_groups)
    return result, inter_result


if __name__ == "__main__":
    extractor = Extractor()
    extractor.device = "cuda:1"

    paths = glob.glob('/home/sds/namnt/FWD/data/test16_txn/*/*.*')
    paths = sorted(paths)
    # images = [cv2.imread(path) for path in paths]
    images = [cv2.imread("/home/sds/namnt/FWD/data/test14_shk/images/Scanning_HK03.PNG")]
    s = time()
    result, inter_result = test_auto_flow(images)

    # doc_groups = [{'doc_type': 'OCR002', 'end_page': 1, 'start_page': 0},
    #               {'doc_type': 'OCR010', 'end_page': 19, 'start_page': 2}]
    # result, inter_result = test_manual_clf_flow(images, doc_groups)
    e = time()
    print('Total processing time:', e - s)

    print(result)

    vis_images = vis_debugging('tmp', inter_result)
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    rq_id = 'tmp'
    save_dir = f'output/debug/{date_string}_{rq_id}'
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(vis_images):
        img = img[:, :, ::-1]
        cv2.imwrite(f'{save_dir}/{i}.jpg', img)
