from typing import Union, overload, List, Optional, Tuple
from PIL import Image
import torch
import numpy as np
import yaml
from pathlib import Path
import mmcv
from sdsvtd import StandaloneYOLOXRunner
from sdsvtr import StandaloneSATRNRunner
from sdsv_dewarp.api import AlignImage

from .utils import ImageReader, chunks, Timer, post_process_recog  # rotate_bbox

# from .utils import jdeskew as deskew
# from externals.deskew.sdsv_dewarp import pdeskew as deskew
# from .utils import deskew
from .dto import Word, Line, Page, Document, Box, WordGroup

# from .word_formation import words_to_lines as words_to_lines
# from .word_formation import wo    rds_to_lines_mmocr as words_to_lines
from .word_formation import words_formation_mmocr_tesseract as word_formation

DEFAULT_SETTING_PATH = str(Path(__file__).parents[1]) + "/settings.yml"


class OcrEngine:
    def __init__(self, settings_file: str = DEFAULT_SETTING_PATH, **kwargs):
        """Warper of text detection and text recognition
        :param settings_file: path to default setting file
        :param kwargs: keyword arguments to overwrite the default settings file
        """
        with open(settings_file) as f:
            # use safe_load instead load
            self._settings = yaml.safe_load(f)
        for k, v in kwargs.items():  # overwrite default settings by keyword arguments
            if k not in self._settings:
                raise ValueError("Invalid setting found in OcrEngine: ", k)
            self._settings[k] = v
        self._ensure_device()
        self._detector = StandaloneYOLOXRunner(**self._settings["detector"])
        self._recognizer = StandaloneSATRNRunner(**self._settings["recognizer"])
        if self._settings["deskew"]["enable"]:
            self._deskewer = AlignImage(
                **{k: v for k, v in self._settings["deskew"].items() if k != "enable"}
            )

    def _disclaimer(self):
        if self._settings["deskew"]["enable"]:
            print(
                "[WARNING]: Deskew is enabled. The bounding boxes prediction may not be aligned with the original image. In case of using these predictions for pseudo-label, turn on save_deskewed option and use the saved deskewed images instead for further proceed."
            )

    def _ensure_device(self):
        if "cuda" in self._settings["device"]:
            if not torch.cuda.is_available():
                print("[WARNING]: CUDA is not available, running with cpu instead")
                self._settings["device"] = "cpu"

    @property
    def version(self):
        return {
            "detector": self._settings["detector"],
            "recognizer": self._settings["recognizer"],
        }

    @property
    def settings(self):
        return self._settings

    # @staticmethod
    # def xyxyc_to_xyxy_c(xyxyc: np.ndarray) -> Tuple[List[list], list]:
    #     '''
    #     convert sdsvtd yoloX detection output to list of bboxes and list of confidences
    #     @param xyxyc: array of shape (n, 5)
    #     '''
    #     xyxy = xyxyc[:, :4].tolist()
    #     confs = xyxyc[:, 4].tolist()
    #     return xyxy, confs
    # -> Tuple[np.ndarray, List[Box]]:

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, bool, float]:
        img_ = img.copy()
        if self._settings["img_size"]:
            img_ = mmcv.imrescale(
                img,
                tuple(self._settings["img_size"]),
                return_scale=False,
                interpolation="bilinear",
                backend="cv2",
            )
        is_blank = False
        if self._settings["deskew"]["enable"]:
            with Timer("deskew"):
                img_, is_blank, angle = self._deskewer(img_)
                return img_, is_blank, angle
        # for i, bbox in enumerate(bboxes):
        #     rotated_bbox = rotate_bbox(bbox, angle, img.shape[:2])
        #     bboxes[i].bbox = rotated_bbox
        return img_, is_blank, 0

    def run_detect(
        self, img: np.ndarray, return_raw: bool = False
    ) -> Tuple[np.ndarray, Union[List[Box], List[list]]]:
        """
        run text detection and return list of xyxyc if return_confidence is True, otherwise return a list of xyxy
        """
        pred_det = self._detector(img)
        if self._settings["detector"]["auto_rotate"]:
            img, pred_det = pred_det
        pred_det = pred_det[0]  # only image at a time
        return (
            (img, pred_det.tolist())
            if return_raw
            else (img, [Box(*xyxyc) for xyxyc in pred_det.tolist()])
        )

    def run_recog(
        self, imgs: List[np.ndarray]
    ) -> Union[List[str], List[Tuple[str, float]]]:
        if len(imgs) == 0:
            return list()
        pred_rec = self._recognizer(imgs)
        return [
            (post_process_recog(word), conf)
            for word, conf in zip(pred_rec[0], pred_rec[1])
        ]

    def read_img(self, img: str) -> np.ndarray:
        return ImageReader.read(img)

    def get_cropped_imgs(
        self, img: np.ndarray, bboxes: Union[List[Box], List[list]]
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        img: np image
        bboxes: list of xyxy
        """
        lcropped_imgs = list()
        mask = list()
        for bbox in bboxes:
            bbox = Box(*bbox) if isinstance(bbox, list) else bbox
            bbox = bbox.get_extend_bbox(self._settings["extend_bbox"])

            bbox.clamp_by_img_wh(img.shape[1], img.shape[0])
            bbox.to_int()
            if not bbox.is_valid():
                mask.append(False)
                continue
            cropped_img = bbox.crop_img(img)
            lcropped_imgs.append(cropped_img)
            mask.append(True)
        return lcropped_imgs, mask

    def read_page(
        self, img: np.ndarray, bboxes: Union[List[Box], List[list]]
    ) -> Union[List[WordGroup], List[Line]]:
        if len(bboxes) == 0:  # no bbox found
            return list()
        with Timer("cropped imgs"):
            lcropped_imgs, mask = self.get_cropped_imgs(img, bboxes)
        with Timer("recog"):
            # batch_mode for efficiency
            pred_recs = self.run_recog(lcropped_imgs)
        with Timer("construct words"):
            lwords = list()
            for i in range(len(pred_recs)):
                if not mask[i]:
                    continue
                text, conf_rec = pred_recs[i][0], pred_recs[i][1]
                bbox = Box(*bboxes[i]) if isinstance(bboxes[i], list) else bboxes[i]
                lwords.append(
                    Word(
                        image=img,
                        text=text,
                        conf_cls=conf_rec,
                        bbox_obj=bbox,
                        conf_detect=bbox._conf,
                    )
                )
        with Timer("word formation"):
            return word_formation(
                lwords, img.shape[1], **self._settings["words_to_lines"]
            )[0]

    # https://stackoverflow.com/questions/48127642/incompatible-types-in-assignment-on-union

    @overload
    def __call__(self, img: Union[str, np.ndarray, Image.Image]) -> Page:
        ...

    @overload
    def __call__(self, img: List[Union[str, np.ndarray, Image.Image]]) -> Document:
        ...

    def __call__(self, img):  # type: ignore #ignoring type before implementing batch_mode
        """
        Accept an image or list of them, return ocr result as a page or document
        """
        with Timer("read image"):
            img = ImageReader.read(img)
        if self._settings["batch_size"] == 1:
            if isinstance(img, list):
                if len(img) == 1:
                    img = img[0]  # in case input type is a 1 page pdf
                else:
                    raise AssertionError(
                        "list input can only be used with batch_mode enabled"
                    )
            img_deskewed, is_blank, angle = self.preprocess(img)

            if is_blank:
                print(
                    "[WARNING]: Blank image detected"
                )  # TODO: should we stop the execution here?
            with Timer("detect"):
                img_deskewed, bboxes = self.run_detect(img_deskewed)
            with Timer("read_page"):
                lsegments = self.read_page(img_deskewed, bboxes)
            return Page(lsegments, img, img_deskewed if angle != 0 else None)
        else:
            # lpages = []
            # # chunks to reduce memory footprint
            # for imgs in chunks(img, self._batch_size):
            #     # pred_dets = self._detector(imgs)
            #     # TEMP: use list comprehension because sdsvtd do not support batch mode of text detection
            #     img = self.preprocess(img)
            #     img, bboxes = self.run_detect(img)
            #     for img_, bboxes_ in zip(imgs, bboxes):
            #         llines = self.read_page(img, bboxes_)
            #         page = Page(llines, img)
            #         lpages.append(page)
            # return Document(lpages)
            raise NotImplementedError("Batch mode is currently not supported")


if __name__ == "__main__":
    img_path = "/mnt/ssd1T/hungbnt/Cello/data/PH/Sea7/Sea_7_1.jpg"
    engine = OcrEngine(device="cuda:0")
    # https://stackoverflow.com/questions/66435480/overload-following-optional-argument
    page = engine(img_path)  # type: ignore
    print(page._word_segments)
