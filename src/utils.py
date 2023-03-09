from PIL import ImageFont, ImageDraw, Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from typing import Generator, Union, List, overload
import glob
from pdf2image import convert_from_path


class ImageReader:
    """
    accept anything, return numpy array image
    """
    supported_ext = [".png", ".jpg", ".jpeg", ".pdf"]

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
    def from_dir(dir_path: str) -> List[np.array]:
        if os.path.isdir(dir_path):
            image_files = list()
            for ext in ImageReader.supported_ext:
                image_files = glob.glob(os.path.join(dir_path, "*" + ext))
            return ImageReader.from_list(image_files)
        else:
            raise NotADirectoryError(dir_path)

    @staticmethod
    def from_str(img_path: str) -> np.ndarray:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        return np.array(Image.open(img_path))

    @staticmethod
    def from_numpy(img_array: np.ndarray) -> np.ndarray:
        return img_array

    @staticmethod
    def from_PIL(img_pil: Image.Image) -> np.ndarray:
        return np.array(img_pil)

    @staticmethod
    def from_list(img_list: List[Union[str, np.ndarray, Image.Image]]) -> List[np.ndarray]:
        return [ImageReader._read(img_path) for img_path in img_list]

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
            return ImageReader.from_numpy(img)
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


def read_ocr_result_from_txt(file_path: str) -> tuple[list, list]:
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


def get_dynamic_params_for_bbox_of_label(text, x1, y1, w, h, img_h, img_w, font):
    font_scale_factor = img_h / (img_w + img_h)
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
