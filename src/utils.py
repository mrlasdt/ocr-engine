from PIL import ImageFont, ImageDraw, Image, ImageOps
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
from typing import Generator, Union, List, overload, Tuple, Callable
import glob
import math
from pathlib import Path
from pdf2image import convert_from_path
from deskew import determine_skew
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate as jrotate


def post_process_recog(text: str) -> str:
    text = text.replace("✪", " ")
    return text


def find_maximum_without_outliers(lst: list[int], threshold: float = 1.):
    '''
    To find the maximum number in a list while excluding its outlier values, you can follow these steps:
    Determine the range within which you consider values as outliers. This can be based on a specific threshold or a statistical measure such as the interquartile range (IQR).
    Iterate through the list and filter out the outlier values based on the defined range. Keep track of the non-outlier values.
    Find the maximum value among the non-outlier values.
    '''
    # Calculate the lower and upper boundaries for outliers
    q1 = np.percentile(lst, 25)
    q3 = np.percentile(lst, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Filter out outlier values
    non_outliers = [x for x in lst if lower_bound <= x <= upper_bound]

    # Find the maximum value among non-outliers
    max_value = max(non_outliers)

    return max_value


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, func: Callable, *args):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"[INFO]: {self.name} took : {self.elapsed_time:.6f} seconds")


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


# def rotate_bbox(bbox: list, angle: float) -> list:
#     # Compute the center point of the bounding box
#     cx = bbox[0] + bbox[2] / 2
#     cy = bbox[1] + bbox[3] / 2

#     # Define the scale factor for the rotated bounding box
#     scale = 1.0  # following the deskew and jdeskew function
#     angle_radian = math.radians(angle)

#     # Obtain the rotation matrix using cv2.getRotationMatrix2D()
#     M = cv2.getRotationMatrix2D((cx, cy), angle_radian, scale)

#     # Apply the rotation matrix to the four corners of the bounding box
#     corners = np.array([[bbox[0], bbox[1]],
#                         [bbox[0] + bbox[2], bbox[1]],
#                         [bbox[0] + bbox[2], bbox[1] + bbox[3]],
#                         [bbox[0], bbox[1] + bbox[3]]], dtype=np.float32)
#     rotated_corners = cv2.transform(np.array([corners]), M)[0]

#     # Compute the bounding box of the rotated corners
#     x = int(np.min(rotated_corners[:, 0]))
#     y = int(np.min(rotated_corners[:, 1]))
#     w = int(np.max(rotated_corners[:, 0]) - np.min(rotated_corners[:, 0]))
#     h = int(np.max(rotated_corners[:, 1]) - np.min(rotated_corners[:, 1]))
#     rotated_bbox = [x, y, w, h]

#     return rotated_bbox

def rotate_bbox(bbox: List[int], angle: float, old_shape: Tuple[int, int]) -> List[int]:
    # https://medium.com/@pokomaru/image-and-bounding-box-rotation-using-opencv-python-2def6c39453
    bbox_ = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
    h, w = old_shape
    cx, cy = (int(w / 2), int(h / 2))

    bbox_tuple = [
        (bbox_[0], bbox_[1]),
        (bbox_[2], bbox_[3]),
        (bbox_[4], bbox_[5]),
        (bbox_[6], bbox_[7]),
    ]  # put x and y coordinates in tuples, we will iterate through the tuples and perform rotation

    rotated_bbox = []

    for i, coord in enumerate(bbox_tuple):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        newW = int((h * sin) + (w * cos))
        newH = int((h * cos) + (w * sin))
        M[0, 2] += (newW / 2) - cx
        M[1, 2] += (newH / 2) - cy
        v = [coord[0], coord[1], 1]
        adjusted_coord = np.dot(M, v)
        rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))
    result = [int(x) for t in rotated_bbox for x in t]
    return [result[i] for i in [0, 1, 2, -1]]  # reformat to xyxy


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = 0.
    try:
        angle = determine_skew(grayscale)
    except Exception:
        pass
    rotated = rotate(image, angle, (0, 0, 0)) if angle else image
    return rotated, angle


def jdeskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    angle = 0.
    try:
        angle = get_angle(image)
    except Exception:
        pass
    # TODO: change resize = True and scale the bounding box
    rotated = jrotate(image, angle, resize=False) if angle else image
    return rotated, angle


class ImageReader:
    """
    accept anything, return numpy array image
    """
    supported_ext = [".png", ".jpg", ".jpeg", ".pdf", ".gif"]

    @staticmethod
    def validate_img_path(img_path: str) -> None:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        if os.path.isdir(img_path):
            raise IsADirectoryError(img_path)
        if not Path(img_path).suffix.lower() in ImageReader.supported_ext:
            raise NotImplementedError("Not supported extension at {}".format(img_path))

    @overload
    @staticmethod
    def read(img: Union[str, np.ndarray, Image.Image]) -> np.ndarray: ...

    @overload
    @staticmethod
    def read(img: List[Union[str, np.ndarray, Image.Image]]) -> List[np.ndarray]: ...

    @overload
    @staticmethod
    def read(img: str) -> List[np.ndarray]: ...  # for pdf or directory

    @staticmethod
    def read(img):
        if isinstance(img, list):
            return ImageReader.from_list(img)
        elif isinstance(img, str) and os.path.isdir(img):
            return ImageReader.from_dir(img)
        elif isinstance(img, str) and img.endswith(".pdf"):
            return ImageReader.from_pdf(img)
        else:
            return ImageReader._read(img)

    @staticmethod
    def from_dir(dir_path: str) -> List[np.ndarray]:
        if os.path.isdir(dir_path):
            image_files = glob.glob(os.path.join(dir_path, "*"))
            return ImageReader.from_list(image_files)
        else:
            raise NotADirectoryError(dir_path)

    @staticmethod
    def from_str(img_path: str) -> np.ndarray:
        ImageReader.validate_img_path(img_path)
        return ImageReader.from_PIL(Image.open(img_path))

    @staticmethod
    def from_np(img_array: np.ndarray) -> np.ndarray:
        return img_array

    @staticmethod
    def from_PIL(img_pil: Image.Image, transpose=True) -> np.ndarray:
        # if img_pil.is_animated:
        #     raise NotImplementedError("Only static images are supported, animated image found")
        if transpose:
            img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        return np.array(img_pil)

    @staticmethod
    def from_list(img_list: List[Union[str, np.ndarray, Image.Image]]) -> List[np.ndarray]:
        limgs = list()
        for img_path in img_list:
            try:
                if isinstance(img_path, str):
                    ImageReader.validate_img_path(img_path)
                limgs.append(ImageReader._read(img_path))
            except (FileNotFoundError, NotImplementedError, IsADirectoryError) as e:
                print("[ERROR]: ", e)
                print("[INFO]: Skipping image {}".format(img_path))
        return limgs

    @staticmethod
    def from_pdf(pdf_path: str, start_page: int = 0, end_page: int = 0) -> List[np.ndarray]:
        pdf_file = convert_from_path(pdf_path)
        if end_page is not None:
            end_page = min(len(pdf_file), end_page + 1)
        limgs = [np.array(pdf_page) for pdf_page in pdf_file[start_page:end_page]]
        return limgs

    @staticmethod
    def _read(img: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, str):
            return ImageReader.from_str(img)
        elif isinstance(img, Image.Image):
            return ImageReader.from_PIL(img)
        elif isinstance(img, np.ndarray):
            return ImageReader.from_np(img)
        else:
            raise ValueError("Invalid img argument type: ", type(img))


def get_name(file_path, ext: bool = True):
    file_path_ = os.path.basename(file_path)
    return file_path_ if ext else os.path.splitext(file_path_)[0]


def construct_file_path(dir, file_path, ext=''):
    '''
    args:
        dir: /path/to/dir
        file_path /example_path/to/file.txt
        ext = '.json'
    return 
        /path/to/dir/file.json
    '''
    return os.path.join(
        dir, get_name(file_path,
                      True)) if ext == '' else os.path.join(
        dir, get_name(file_path,
                      False)) + ext


def chunks(lst: list, n: int) -> Generator:
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_ocr_result_from_txt(file_path: str) -> Tuple[list, list]:
    '''
    return list of bounding boxes, list of words
    '''
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    boxes, words = [], []
    for line in lines:
        if line == "":
            continue
        x1, y1, x2, y2, text = line.split("\t")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if text and text != " ":
            words.append(text)
            boxes.append((x1, y1, x2, y2))
    return boxes, words


def get_xyxywh_base_on_format(bbox, format):
    if format == "xywh":
        x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2, y2 = x1 + w, y1 + h
    elif format == "xyxy":
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
    else:
        raise NotImplementedError("Invalid format {}".format(format))
    return (x1, y1, x2, y2, w, h)


def get_dynamic_params_for_bbox_of_label(text, x1, y1, w, h, img_h, img_w, font, font_scale_offset=1):
    font_scale_factor = img_h / (img_w + img_h) * font_scale_offset
    font_scale = w / (w + h) * font_scale_factor  # adjust font scale by width height
    thickness = int(font_scale_factor) + 1
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    text_offset_x = x1
    text_offset_y = y1 - thickness
    box_coords = ((text_offset_x, text_offset_y + 1), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    return (font_scale, thickness, text_height, box_coords)


def visualize_bbox_and_label(
        img, bboxes, texts, bbox_color=(200, 180, 60),
        text_color=(0, 0, 0),
        format="xyxy", is_vnese=False, draw_text=True):
    ori_img_type = type(img)
    if is_vnese:
        img = Image.fromarray(img) if ori_img_type is np.ndarray else img
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        font_pil_str = "fonts/arial.ttf"
        font_cv2 = cv2.FONT_HERSHEY_SIMPLEX
    else:
        img_h, img_w = img.shape[0], img.shape[1]
        font_cv2 = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(bboxes)):
        text = texts[i]  # text = "{}: {:.0f}%".format(LABELS[classIDs[i]], confidences[i]*100)
        x1, y1, x2, y2, w, h = get_xyxywh_base_on_format(bboxes[i], format)
        font_scale, thickness, text_height, box_coords = get_dynamic_params_for_bbox_of_label(
            text, x1, y1, w, h, img_h, img_w, font=font_cv2)
        if is_vnese:
            font_pil = ImageFont.truetype(font_pil_str, size=text_height)  # type: ignore
            fdraw_text = draw.text  # type: ignore
            fdraw_bbox = draw.rectangle  # type: ignore
            # Pil use different coordinate => y = y+thickness = y-thickness + 2*thickness
            arg_text = ((box_coords[0][0], box_coords[1][1]), text)
            kwarg_text = {"font": font_pil, "fill": text_color, "width": thickness}
            arg_rec = ((x1, y1, x2, y2),)
            kwarg_rec = {"outline": bbox_color, "width": thickness}
            arg_rec_text = ((box_coords[0], box_coords[1]),)
            kwarg_rec_text = {"fill": bbox_color, "width": thickness}
        else:
            # cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
            # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(50, 0,0), thickness=thickness)
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            fdraw_text = cv2.putText
            fdraw_bbox = cv2.rectangle
            arg_text = (img, text, box_coords[0])
            kwarg_text = {"fontFace": font_cv2, "fontScale": font_scale, "color": text_color, "thickness": thickness}
            arg_rec = (img, (x1, y1), (x2, y2))
            kwarg_rec = {"color": bbox_color, "thickness": thickness}
            arg_rec_text = (img, box_coords[0], box_coords[1])
            kwarg_rec_text = {"color": bbox_color, "thickness": cv2.FILLED}
        # draw a bounding box rectangle and label on the img
        fdraw_bbox(*arg_rec, **kwarg_rec)  # type: ignore
        if draw_text:
            fdraw_bbox(*arg_rec_text, **kwarg_rec_text)  # type: ignore
            fdraw_text(*arg_text, **kwarg_text)  # type: ignore   # text have to put in front of rec_text
    return np.array(img) if ori_img_type is np.ndarray and is_vnese else img
