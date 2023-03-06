# %% #############################################################################################################################################################################################################################################################
import cv2
import sdsvtd
import sdsvtr 
print(sdsvtd.__hub_available_versions__)
print(sdsvtr.__hub_available_versions__)

#%%
from sdsvtd import StandaloneYOLOXRunner
from sdsvtr import StandaloneSATRNRunner
detector = StandaloneYOLOXRunner(version="yolox-s-general-text-pretrain-20221226", device="cuda:1")
recognitor = StandaloneSATRNRunner(version="satrn-lite-general-pretrain-20230106", return_confident=True, device="cuda:1")
#%%
img_path = "/mnt/ssd500/hungbnt/Cello/data/PH/Sea7/Sea_7_1.jpg"
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#%%
# detection_out = detector([img, img])\
detection_out = detector(img)
# print(detecti/on_out)
detection_out[0].shape

#%%
detection_out[0].tolist()
#%%
import numpy as np
np.array(detection_out[0])[:,:4].tolist()
#%%
np.array(detection_out[0])[:,4].tolist()

#%%
for out in detection_out:
    for bbox in out:
        recognition_out = recognitor(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        print(recognition_out)
cv2.imwrite('table_img_detection.png', img)
# %% #############################################################################################################################################################################################################################################################

# recognitor_with_conf = StandaloneSATRNRunner(version="satrn-lite-general-pretrain-20230106", return_confident=True, device="cuda:1")
# recognitor_with_conf(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
# %%
import numpy as np
print(detection_out.shape)
np.vstack(detection_out).shape
# %%
