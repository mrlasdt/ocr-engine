import numpy as np
from typing import Optional, List, Union
import cv2
from PIL import Image
from .utils import visualize_bbox_and_label


class Box:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float = -1., label: str = ""):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._conf = conf
        self._label = label

    def __repr__(self) -> str:
        return str(self.bbox)

    def __str__(self) -> str:
        return str(self.bbox)

    def get(self, return_confidence=False) -> Union[list[int], list[Union[float, int]]]:
        return self.bbox if not return_confidence else self.xyxyc

    def __getitem__(self, key):
        return self.bbox[key]

    @property
    def width(self):
        return max(self._x2 - self._x1, -1)

    @property
    def height(self):
        return max(self._y2 - self._y1, -1)

    @property
    def bbox(self) -> list[int]:
        return [self._x1, self._y1, self._x2, self._y2]

    @bbox.setter
    def bbox(self, bbox_: list[int]):
        self._x1, self._y1, self._x2, self._y2 = bbox_

    @property
    def xyxyc(self) -> list[Union[float, int]]:
        return [self._x1, self._y1, self._x2, self._y2, self._conf]

    @staticmethod
    def normalize_bbox(bbox: list[int]) -> list[int]:
        return [int(b) for b in bbox]

    def to_int(self):
        self._x1, self._y1, self._x2, self._y2 = self.normalize_bbox([self._x1, self._y1, self._x2, self._y2])
        return self

    @staticmethod
    def clamp_bbox_by_img_wh(bbox: list, width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        x1 = min(max(0, x1), width)
        x2 = min(max(0, x2), width)
        y1 = min(max(0, y1), height)
        y2 = min(max(0, y2), height)
        return [x1, y1, x2, y2]

    def clamp_by_img_wh(self, width: int, height: int):
        self._x1, self._y1, self._x2, self._y2 = self.clamp_bbox_by_img_wh(
            [self._x1, self._y1, self._x2, self._y2], width, height)
        return self

    @staticmethod
    def extend_bbox(bbox: list, margin: list):  # -> Self (python3.11)
        margin_l, margin_t, margin_r, margin_b = margin
        l, t, r, b = bbox  # left, top, right, bottom
        t = t - (b - t) * margin_t
        b = b + (b - t) * margin_b
        l = l - (r - l) * margin_l
        r = r + (r - l) * margin_r
        return [l, t, r, b]

    def get_extend_bbox(self, margin: list):
        extended_bbox = self.extend_bbox(self.bbox, margin)
        return Box(*extended_bbox, label=self._label)

    @staticmethod
    def bbox_is_valid(bbox: list[int]) -> bool:
        if bbox == [-1, -1, -1, -1]:
            raise ValueError("Empty bounding box found")
        l, t, r, b = bbox  # left, top, right, bottom
        return True if (b - t) * (r - l) > 0 else False

    def is_valid(self) -> bool:
        return self.bbox_is_valid(self.bbox)

    @staticmethod
    def crop_img_by_bbox(img: np.ndarray, bbox: list) -> np.ndarray:
        l, t, r, b = bbox
        return img[t:b, l:r]

    def crop_img(self, img: np.ndarray) -> np.ndarray:
        return self.crop_img_by_bbox(img, self.bbox)


class Word:
    def __init__(
        self,
        image=None,
        text="",
        conf_cls=-1.,
        bbox_obj: Box = Box(-1, -1, -1, -1),
        conf_detect=-1.,
        kie_label="",
    ):
        # self.type = "word"
        self._text = text
        self._image = image
        self._conf_det = conf_detect
        self._conf_cls = conf_cls
        # [left, top,right,bot] coordinate of top-left and bottom-right point
        self._bbox_obj = bbox_obj
        # self.word_id = 0  # id of word
        # self.word_group_id = 0  # id of word_group which instance belongs to
        # self.line_id = 0  # id of line which instance belongs to
        # self.paragraph_id = 0  # id of line which instance belongs to
        self._kie_label = kie_label

    @property
    def bbox(self) -> list[int]:
        return self._bbox_obj.bbox

    @property
    def text(self) -> str:
        return self._text

    @property
    def height(self):
        return self._bbox_obj.height

    @property
    def width(self):
        return self._bbox_obj.width

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def is_valid(self) -> bool:
        return self._bbox_obj.is_valid()

    # def is_special_word(self):
    #     if not self._text:
    #         raise ValueError("Cannot validatie size of empty bounding box")

    #     # if len(text) > 7:
    #     #     return True
    #     if len(self._text) >= 7:
    #         no_digits = sum(c.isdigit() for c in text)
    #         return no_digits / len(text) >= 0.3

    #     return False


class WordGroup:
    def __init__(self, list_words: List[Word] = list(),
                 text: str = '', boundingbox: Box = Box(-1, -1, -1, -1), conf_cls: float = -1, conf_det: float = -1):
        # self.type = "word_group"
        self._list_words = list_words  # dict of word instances
        # self.word_group_id = 0  # word group id
        # self.line_id = 0  # id of line which instance belongs to
        # self.paragraph_id = 0  # id of paragraph which instance belongs to
        self._text = text
        self._bbox_obj = boundingbox
        self._kie_label = ""
        self._conf_cls = conf_cls
        self._conf_det = conf_det

    @property
    def bbox(self) -> list[int]:
        return self._bbox_obj.bbox

    @property
    def text(self) -> str:
        return self._text

    @property
    def list_words(self) -> list[Word]:
        return self._list_words

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    # def add_word(self, word: Word):  # add a word instance to the word_group
    #     if word._text != "✪":
    #         for w in self._list_words:
    #             if word.word_id == w.word_id:
    #                 print("Word id collision")
    #                 return False
    #         word.word_group_id = self.word_group_id  #
    #         word.line_id = self.line_id
    #         word.paragraph_id = self.paragraph_id
    #         self._list_words.append(word)
    #         self._text += " " + word._text
    #         if self.bbox_obj == [-1, -1, -1, -1]:
    #             self.bbox_obj = word._bbox_obj
    #         else:
    #             self.bbox_obj = [
    #                 min(self.bbox_obj[0], word._bbox_obj[0]),
    #                 min(self.bbox_obj[1], word._bbox_obj[1]),
    #                 max(self.bbox_obj[2], word._bbox_obj[2]),
    #                 max(self.bbox_obj[3], word._bbox_obj[3]),
    #             ]
    #         return True
    #     else:
    #         return False

    # def update_word_group_id(self, new_word_group_id):
    #     self.word_group_id = new_word_group_id
    #     for i in range(len(self._list_words)):
    #         self._list_words[i].word_group_id = new_word_group_id

    # def update_kie_label(self):
    #     list_kie_label = [word._kie_label for word in self._list_words]
    #     dict_kie = dict()
    #     for label in list_kie_label:
    #         if label not in dict_kie:
    #             dict_kie[label] = 1
    #         else:
    #             dict_kie[label] += 1
    #     total = len(list(dict_kie.values()))
    #     max_value = max(list(dict_kie.values()))
    #     list_keys = list(dict_kie.keys())
    #     list_values = list(dict_kie.values())
    #     self.kie_label = list_keys[list_values.index(max_value)]

    # def update_text(self):  # update text after changing positions of words in list word
    #     text = ""
    #     for word in self._list_words:
    #         text += " " + word._text
    #     self._text = text


class Line:
    def __init__(self, list_word_groups: List[WordGroup] = [],
                 text: str = '', boundingbox: Box = Box(-1, -1, -1, -1), conf_cls: float = -1, conf_det: float = -1):
        # self.type = "line"
        self._list_word_groups = list_word_groups  # list of Word_group instances in the line
        # self.line_id = 0  # id of line in the paragraph
        # self.paragraph_id = 0  # id of paragraph which instance belongs to
        self._text = text
        self._bbox_obj = boundingbox
        self._conf_cls = conf_cls
        self._conf_det = conf_det

    @property
    def bbox(self) -> list[int]:
        return self._bbox_obj.bbox

    @property
    def text(self) -> str:
        return self._text

    @property
    def list_word_groups(self) -> List[WordGroup]:
        return self._list_word_groups

    @property
    def list_words(self) -> list[Word]:
        return [word for word_group in self._list_word_groups for word in word_group.list_words]

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    # def add_group(self, word_group: WordGroup):  # add a word_group instance
    #     if word_group._list_words is not None:
    #         for wg in self.list_word_groups:
    #             if word_group.word_group_id == wg.word_group_id:
    #                 print("Word_group id collision")
    #                 return False

    #         self.list_word_groups.append(word_group)
    #         self.text += word_group._text
    #         word_group.paragraph_id = self.paragraph_id
    #         word_group.line_id = self.line_id

    #         for i in range(len(word_group._list_words)):
    #             word_group._list_words[
    #                 i
    #             ].paragraph_id = self.paragraph_id  # set paragraph_id for word
    #             word_group._list_words[i].line_id = self.line_id  # set line_id for word
    #         return True
    #     return False

    # def update_line_id(self, new_line_id):
    #     self.line_id = new_line_id
    #     for i in range(len(self.list_word_groups)):
    #         self.list_word_groups[i].line_id = new_line_id
    #         for j in range(len(self.list_word_groups[i]._list_words)):
    #             self.list_word_groups[i]._list_words[j].line_id = new_line_id

    # def merge_word(self, word):  # word can be a Word instance or a Word_group instance
    #     if word.text != "✪":
    #         if self.boundingbox == [-1, -1, -1, -1]:
    #             self.boundingbox = word.boundingbox
    #         else:
    #             self.boundingbox = [
    #                 min(self.boundingbox[0], word.boundingbox[0]),
    #                 min(self.boundingbox[1], word.boundingbox[1]),
    #                 max(self.boundingbox[2], word.boundingbox[2]),
    #                 max(self.boundingbox[3], word.boundingbox[3]),
    #             ]
    #         self.list_word_groups.append(word)
    #         self.text += " " + word.text
    #         return True
    #     return False

    # def __cal_ratio(self, top1, bottom1, top2, bottom2):
    #     sorted_vals = sorted([top1, bottom1, top2, bottom2])
    #     intersection = sorted_vals[2] - sorted_vals[1]
    #     min_height = min(bottom1 - top1, bottom2 - top2)
    #     if min_height == 0:
    #         return -1
    #     ratio = intersection / min_height
    #     return ratio

    # def __cal_ratio_height(self, top1, bottom1, top2, bottom2):

    #     height1, height2 = top1 - bottom1, top2 - bottom2
    #     ratio_height = float(max(height1, height2)) / float(min(height1, height2))
    #     return ratio_height

    # def in_same_line(self, input_line, thresh=0.7):
    #     # calculate iou in vertical direction
    #     _, top1, _, bottom1 = self.boundingbox
    #     _, top2, _, bottom2 = input_line.boundingbox

    #     ratio = self.__cal_ratio(top1, bottom1, top2, bottom2)
    #     ratio_height = self.__cal_ratio_height(top1, bottom1, top2, bottom2)

    #     if (
    #         (top2 <= top1 <= bottom2) or (top1 <= top2 <= bottom1)
    #         and ratio >= thresh
    #         and (ratio_height < 2)
    #     ):
    #         return True
    #     return False


# class Paragraph:
#     def __init__(self, id=0, lines=None):
#         self.list_lines = lines if lines is not None else []  # list of all lines in the paragraph
#         self.paragraph_id = id  # index of paragraph in the ist of paragraph
#         self.text = ""
#         self.boundingbox = [-1, -1, -1, -1]

#     @property
#     def bbox(self):
#         return self.boundingbox

#     def __repr__(self) -> str:
#         return self.text

#     def __str__(self) -> str:
#         return self.text

#     def add_line(self, line: Line):  # add a line instance
#         if line.list_word_groups is not None:
#             for l in self.list_lines:
#                 if line.line_id == l.line_id:
#                     print("Line id collision")
#                     return False
#             for i in range(len(line.list_word_groups)):
#                 line.list_word_groups[
#                     i
#                 ].paragraph_id = (
#                     self.paragraph_id
#                 )  # set paragraph id for every word group in line
#                 for j in range(len(line.list_word_groups[i]._list_words)):
#                     line.list_word_groups[i]._list_words[
#                         j
#                     ].paragraph_id = (
#                         self.paragraph_id
#                     )  # set paragraph id for every word in word groups
#             line.paragraph_id = self.paragraph_id  # set paragraph id for line
#             self.list_lines.append(line)  # add line to paragraph
#             self.text += " " + line.text
#             return True
#         else:
#             return False

#     def update_paragraph_id(
#         self, new_paragraph_id
#     ):  # update new paragraph_id for all lines, word_groups, words inside paragraph
#         self.paragraph_id = new_paragraph_id
#         for i in range(len(self.list_lines)):
#             self.list_lines[
#                 i
#             ].paragraph_id = new_paragraph_id  # set new paragraph_id for line
#             for j in range(len(self.list_lines[i].list_word_groups)):
#                 self.list_lines[i].list_word_groups[
#                     j
#                 ].paragraph_id = new_paragraph_id  # set new paragraph_id for word_group
#                 for k in range(len(self.list_lines[i].list_word_groups[j].list_words)):
#                     self.list_lines[i].list_word_groups[j].list_words[
#                         k
#                     ].paragraph_id = new_paragraph_id  # set new paragraph id for word
#         return True


class Page:
    def __init__(
            self, word_segments: Union[List[WordGroup],
                                       List[Line]],
            image: np.ndarray, deskewed_image: Optional[np.ndarray] = None) -> None:
        self._word_segments = word_segments
        self._image = image
        self._deskewed_image = deskewed_image
        self._drawed_image: Optional[np.ndarray] = None

    @property
    def word_segments(self):
        return self._word_segments

    @property
    def list_words(self) -> list[Word]:
        return [word for word_segment in self._word_segments for word in word_segment.list_words]

    @property
    def image(self):
        return self._image

    @property
    def PIL_image(self):
        return Image.fromarray(self._image)

    @property
    def drawed_image(self):
        return self._drawed_image

    def visualize_bbox_and_label(self, **kwargs: dict):
        if self._drawed_image is not None:
            return self._drawed_image
        bboxes = list()
        texts = list()
        for word in self.list_words:
            bboxes.append([int(float(b)) for b in word.bbox])
            texts.append(word._text)
        img = visualize_bbox_and_label(
            self._deskewed_image if self._deskewed_image is not None else self._image, bboxes, texts, **kwargs)
        self._drawed_image = img
        return self._drawed_image

    def save_img(self, save_path: str, **kwargs: dict) -> None:
        img = self.visualize_bbox_and_label(**kwargs)
        cv2.imwrite(save_path, img)

    def write_to_file(self, mode: str, save_path: str) -> None:
        f = open(save_path, "w+", encoding="utf-8")
        for word_segment in self._word_segments:
            if mode == 'segment':
                xmin, ymin, xmax, ymax = word_segment.bbox
                f.write("{}\t{}\t{}\t{}\t{}\n".format(xmin, ymin, xmax, ymax, word_segment._text))
            elif mode == "word":
                for word in word_segment.list_words:
                    # xmin, ymin, xmax, ymax = word.bbox
                    xmin, ymin, xmax, ymax = [int(float(b)) for b in word.bbox]
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(xmin, ymin, xmax, ymax, word._text))
            else:
                raise NotImplementedError("Unknown mode: {}".format(mode))
        f.close()


class Document:
    def __init__(self, lpages: List[Page]) -> None:
        self.lpages = lpages
