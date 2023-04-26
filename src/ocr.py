from typing import Union, overload, List, Optional, Tuple
from PIL import Image
import torch
import numpy as np
import yaml
from pathlib import Path
import mmcv
from sdsvtd import StandaloneYOLOXRunner
from sdsvtr import StandaloneSATRNRunner
from .utils import ImageReader, chunks, rotate_bbox, Timer
# from .utils import jdeskew as deskew
# from externals.deskew.sdsv_dewarp import pdeskew as deskew
from .utils import deskew
from .dto import Word, Line, Page, Document, Box
# from .word_formation import words_to_lines as words_to_lines
# from .word_formation import wo    rds_to_lines_mmocr as words_to_lines
from .word_formation import words_to_lines_tesseract as words_to_lines
DEFAULT_SETTING_PATH = str(Path(__file__).parents[1]) + "/settings.yml"


class OcrEngine:
    def __init__(self, settings_file: str = DEFAULT_SETTING_PATH, **kwargs: dict):
        """ Warper of text detection and text recognition
        :param settings_file: path to default setting file
        :param kwargs: keyword arguments to overwrite the default settings file
        """

        with open(settings_file) as f:
            # use safe_load instead load
            self.__settings = yaml.safe_load(f)
        for k, v in kwargs.items():  # overwrite default settings by keyword arguments
            if k not in self.__settings:
                raise ValueError("Invalid setting found in OcrEngine: ", k)
            self.__settings[k] = v

        if "cuda" in self.__settings["device"]:
            if not torch.cuda.is_available():
                print("[WARNING]: CUDA is not available, running with cpu instead")
                self.__settings["device"] = "cpu"
        self._detector = StandaloneYOLOXRunner(
            version=self.__settings["detector"],
            device=self.__settings["device"],
            auto_rotate=self.__settings["auto_rotate"],
            rotator_version=self.__settings["rotator_version"])
        self._recognizer = StandaloneSATRNRunner(
            version=self.__settings["recognizer"],
            return_confident=True, device=self.__settings["device"])
        # extend the bbox to avoid losing accent mark in vietnames, if using ocr for only english, disable it
        self._do_extend_bbox = self.__settings["do_extend_bbox"]
        # left, top, right, bottom"]
        self._margin_bbox = self.__settings["margin_bbox"]
        self._batch_mode = self.__settings["batch_mode"]
        self._batch_size = self.__settings["batch_size"]
        self._deskew = self.__settings["deskew"]
        self._img_size = self.__settings["img_size"]
        self.__version__ = {
            "detector": self.__settings["detector"],
            "recognizer": self.__settings["recognizer"],
        }

    @property
    def version(self):
        return self.__version__

    @property
    def settings(self):
        return self.__settings

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
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img_ = img.copy()
        if self.__settings["img_size"]:
            img_ = mmcv.imrescale(
                img, tuple(self.__settings["img_size"]),
                return_scale=False, interpolation='bilinear', backend='cv2')
        if self._deskew:
            with Timer("deskew"):
                img_, angle = deskew(img_)
        # for i, bbox in enumerate(bboxes):
        #     rotated_bbox = rotate_bbox(bbox[:], angle, img.shape[:2])
        #     bboxes[i].bbox = rotated_bbox
        return img_  # , bboxes

    def run_detect(self, img: np.ndarray, return_raw: bool = False) -> Tuple[np.ndarray, Union[List[Box], List[list]]]:
        '''
        run text detection and return list of xyxyc if return_confidence is True, otherwise return a list of xyxy
        '''
        pred_det = self._detector(img)
        if self.__settings["auto_rotate"]:
            img, pred_det = pred_det
        pred_det = pred_det[0]  # only image at a time
        return (img, pred_det.tolist()) if return_raw else (img, [Box(*xyxyc) for xyxyc in pred_det.tolist()])

    def run_recog(self, imgs: List[np.ndarray]) -> Union[List[str], List[Tuple[str, float]]]:
        if len(imgs) == 0:
            return list()
        pred_rec = self._recognizer(imgs)
        return [(word, conf) for word, conf in zip(pred_rec[0], pred_rec[1])]

    def read_img(self, img: str) -> np.ndarray:
        return ImageReader.read(img)

    def get_cropped_imgs(self, img: np.ndarray, bboxes: List[Union[Box, list]]) -> Tuple[List[np.ndarray], List[bool]]:
        """
        img: np image
        bboxes: list of xyxy
        """
        lcropped_imgs = list()
        mask = list()
        for bbox in bboxes:
            bbox = Box(*bbox) if isinstance(bbox, list) else bbox
            bbox = bbox.get_extend_bbox(
                self._margin_bbox) if self._do_extend_bbox else bbox
            bbox.clamp_by_img_wh(img.shape[1], img.shape[0])
            bbox.to_int()
            if not bbox.is_valid():
                mask.append(False)
                continue
            cropped_img = bbox.crop_img(img)
            lcropped_imgs.append(cropped_img)
            mask.append(True)
        return lcropped_imgs, mask

    def read_page(self, img: np.ndarray, bboxes: List[Union[Box, list]]) -> List[Line]:
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
                bbox = Box(*bboxes[i]) if isinstance(bboxes[i],
                                                     list) else bboxes[i]
                lwords.append(Word(
                    image=img, text=text, conf_cls=conf_rec, bndbox=bbox, conf_detect=bbox.conf))
        with Timer("words to lines"):
            return words_to_lines(
                lwords, **self.__settings["words_to_lines"])[0]

    # https://stackoverflow.com/questions/48127642/incompatible-types-in-assignment-on-union

    @overload
    def __call__(self, img: Union[str, np.ndarray, Image.Image]) -> Page: ...

    @overload
    def __call__(
        self, img: List[Union[str, np.ndarray, Image.Image]]) -> Document: ...

    def __call__(self, img):
        """
        Accept an image or list of them, return ocr result as a page or document
        """
        with Timer("read image"):
            img = ImageReader.read(img)
        if not self._batch_mode:
            if isinstance(img, list):
                if len(img) == 1:
                    img = img[0]  # in case input type is a 1 page pdf
                else:
                    raise AssertionError(
                        "list input can only be used with batch_mode enabled")
            img = self.preprocess(img)
            with Timer("detect"):
                img, bboxes = self.run_detect(img)
            with Timer("read_page"):
                llines = self.read_page(img, bboxes)
            return Page(llines, img)
        else:
            lpages = []
            # chunks to reduce memory footprint
            for imgs in chunks(img, self._batch_size):
                # pred_dets = self._detector(imgs)
                # TEMP: use list comprehension because sdsvtd do not support batch mode of text detection
                img = self.preprocess(img)
                img, bboxes = self.run_detect(img)
                for img_, bboxes_ in zip(imgs, bboxes):
                    llines = self.read_page(img, bboxes_)
                    page = Page(llines, img)
                    lpages.append(page)
            return Document(lpages)


if __name__ == "__main__":
    img_path = "/mnt/ssd1T/hungbnt/Cello/data/PH/Sea7/Sea_7_1.jpg"
    engine = OcrEngine(device="cuda:0", return_confidence=True)
    # https://stackoverflow.com/questions/66435480/overload-following-optional-argument
    page = engine(img_path)  # type: ignore
    print(page.__llines)
