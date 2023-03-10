from typing import Union, overload, List, Optional
from PIL import Image
import torch
import numpy as np
import yaml
from pathlib import Path
DEFAULT_SETTING_PATH = str(Path(__file__).parents[1]) + "/settings.yml"

from sdsvtd import StandaloneYOLOXRunner
from sdsvtr import StandaloneSATRNRunner
# print(sdsvtr.__hub_available_versions__)
# print(sdsvtd.__hub_available_versions__)

from .utils import ImageReader, chunks
from .dto import Word, Line, Page, Document, Box
from .word_formation import words_to_lines


class OcrEngine:
    def __init__(self, settings_file: str = DEFAULT_SETTING_PATH, **kwargs: dict):
        """ Warper of text detection and text recognition
        :param settings_file: path to default setting file
        :param kwargs: keyword arguments to overwrite the default settings file
        """

        with open(settings_file) as f:
            # use safe_load instead load
            self.settings = yaml.safe_load(f)
        for k, v in kwargs.items():  # overwrite default settings by keyword arguments
            if k not in self.settings:
                raise ValueError("Invalid setting found in OcrEngine: ", k)
            setattr(self, k, v)

        if "cuda" in self.settings["device"]:
            if not torch.cuda.is_available():
                print("[WARNING]: CUDA is not available, running with cpu instead")
                device = "cpu"
            else:
                device = self.settings["device"]
        self.device = device
        self._detector = StandaloneYOLOXRunner(
            version=self.settings["detector"],
            device=device, auto_rotate=self.settings["auto_rotate"])
        self._recognizer = StandaloneSATRNRunner(
            version=self.settings["recognizer"],
            return_confident=True, device=device)
        # extend the bbox to avoid losing accent mark in vietnames, if using ocr for only english, disable it
        self.do_extend_bbox = self.settings["do_extend_bbox"]
        self.margin_bbox = self.settings["margin_bbox"]  # left, top, right, bottom"]
        self.batch_mode = self.settings["batch_mode"]
        self.batch_size = self.settings["batch_size"]
        self.__version__ = {
            "detector": self.settings["detector"],
            "recognizer": self.settings["recognizer"],
        }

    @property
    def version(self):
        return self.__version__

    # @staticmethod
    # def xyxyc_to_xyxy_c(xyxyc: np.ndarray) -> tuple[list[list], list]:
    #     '''
    #     convert sdsvtd yoloX detection output to list of bboxes and list of confidences
    #     @param xyxyc: array of shape (n, 5)
    #     '''
    #     xyxy = xyxyc[:, :4].tolist()
    #     confs = xyxyc[:, 4].tolist()
    #     return xyxy, confs

    def run_detect(self, img: np.ndarray, return_raw: bool = False) -> list[Box]:
        '''
        run text detection and return list of xyxyc if return_confidence is True, otherwise return a list of xyxy
        '''
        pred_det = self._detector(img)
        if self.settings["auto_rotate"]:
            img, pred_det = pred_det
        pred_det = pred_det[0]  # only image at a time
        return (img, pred_det.tolist()) if return_raw else (img, [Box(*xyxyc) for xyxyc in pred_det.tolist()])

    def run_recog(self, imgs: list[np.ndarray]) -> Union[list[str], list[list[str, float]]]:
        pred_rec = self._recognizer(imgs)
        return [[word, conf] for word, conf in zip(pred_rec[0], pred_rec[1])]

    def read_img(self, img: str) -> np.ndarray:
        return ImageReader.read(img)

    def get_cropped_imgs(self, img: np.ndarray, bboxes: list[Union[Box, list]]) -> tuple[list[np.ndarray], list[bool]]:
        """
        img: np image
        bboxes: list of xyxy
        """
        lcropped_imgs = list()
        mask = list()
        for bbox in bboxes:
            bbox = Box(*bbox) if isinstance(bbox, list) else bbox
            bbox = bbox.get_extend_bbox(self.margin_bbox) if self.do_extend_bbox else bbox
            bbox.clamp_by_img_wh(img.shape[1], img.shape[0])
            bbox.normalize()
            if not bbox.is_valid():
                mask.append(False)
                continue
            cropped_img = bbox.crop_img(img)
            lcropped_imgs.append(cropped_img)
            mask.append(True)
        return lcropped_imgs, mask

    def read_page(self, img: np.ndarray, bboxes: list[Union[Box, list]]) -> list[Line]:
        if len(bboxes) == 0:  # no bbox found
            return list()
        lcropped_imgs, mask = self.get_cropped_imgs(img, bboxes)
        pred_recs = self.run_recog(lcropped_imgs)  # batch_mode for efficiency
        lwords = list()
        for i in range(len(pred_recs)):
            if not mask[i]:
                continue
            text, conf_rec = pred_recs[i][0], pred_recs[i][1]
            bbox = Box(*bboxes[i]) if isinstance(bboxes[i], list) else bboxes[i]
            lwords.append(Word(image=img, text=text, conf_cls=conf_rec, bndbox=bbox, conf_detect=bbox.conf))
        return words_to_lines(lwords)[0]

    # https://stackoverflow.com/questions/48127642/incompatible-types-in-assignment-on-union

    @overload
    def __call__(self, img: Union[str, np.ndarray, Image.Image]) -> Page: ...

    @overload
    def __call__(self, img: List[Union[str, np.ndarray, Image.Image]]) -> Document: ...

    def __call__(self, img):
        """
        Accept an image or list of them, return ocr result as a page or document
        """
        img = ImageReader.read(img)
        if not self.batch_mode:
            if isinstance(img, list):
                raise AssertionError("list input can only be used with batch_mode enabled")
            img, bboxes = self.run_detect(img)
            llines = self.read_page(img, bboxes)
            return Page(llines, img)
        else:
            lpages = []
            for imgs in chunks(img, self.batch_size):  # chunks to reduce memory footprint
                # pred_dets = self._detector(imgs)
                # TEMP: use list comprehension because sdsvtd do not support batch mode of text detection
                img, bboxes = self.run_detect(img)
                for img_, bboxes_ in zip(imgs, bboxes):
                    llines = self.read_page(img_, bboxes_)
                    page = Page(llines, img)
                    lpages.append(page)
            return Document(lpages)


if __name__ == "__main__":
    img_path = "/mnt/ssd500/hungbnt/Cello/data/PH/Sea7/Sea_7_1.jpg"
    engine = OcrEngine(device="cuda:0", return_confidence=True)
    # https://stackoverflow.com/questions/66435480/overload-following-optional-argument
    page = engine(img_path)  # type: ignore
    print(page.__llines)
