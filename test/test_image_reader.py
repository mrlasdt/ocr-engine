#%%
from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix()) 
#%%

from src.utils import ImageReader
img_path = "/mnt/ssd1T/hungbnt/DocumentClassification/data/OCR040_043/OCR048/Scanning_HK06.gif"
img = ImageReader.read(img_path)

import matplotlib.pyplot as plt
plt.imshow(img)

#%%

from PIL import Image
import numpy as np
img_pil = Image.open(img_path)
if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
img_cv2 = np.array(img_pil)
plt.imshow(img_pil)
#%%
plt.imshow(img_cv2)

# %%
# import cv2
# img_cv2 = cv2.imread(img_path)
# print(img_cv2)
# cv2.imshow("img_cv2",img_cv2)
# # %% read gif
# import cv2

# # Read the GIF image
# image_path = "/mnt/ssd1T/hungbnt/DocumentClassification/data/OCR040_043/OCR048/Scanning_HK06.gif"  # Replace with the path to your GIF image file
# image = cv2.VideoCapture(image_path)

# # Check if the image was successfully read
# if image.isOpened():
#     frame_count = 0
#     while True:
#         # Read each frame of the GIF image
#         ret, frame = image.read()

#         # Break the loop if the frame is not available
#         if not ret:
#             break

#         # Save the frame as a PNG file
#         output_path = f"frame_{frame_count}.png"
#         cv2.imwrite(output_path, frame)

#         frame_count += 1

#     # Release the capture
#     image.release()
# else:
#     print("Failed to read the image.")



# # %%
# from PIL import Image
# img_pil = Image.open(image_path)
# import matplotlib.pyplot as plt
# plt.imshow(img_pil)
# # %%
# from PIL import ImageOps
# import numpy as np
# img_reader = ImageOps.exif_transpose(img_pil)
# img_reader = np.array(img_pil)
# import matplotlib.pyplot as plt
# plt.imshow(img_reader)
# # %%
