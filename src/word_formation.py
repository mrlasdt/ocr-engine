from builtins import dict
from .dto import Word, Line, Word_group, Box
import numpy as np
from typing import Optional, List, Tuple
MIN_IOU_HEIGHT = 0.7
MIN_WIDTH_LINE_RATIO = 0.05


def resize_to_original(
    boundingbox, scale
):  # resize coordinates to match size of original image
    left, top, right, bottom = boundingbox
    left *= scale[1]
    right *= scale[1]
    top *= scale[0]
    bottom *= scale[0]
    return [left, top, right, bottom]


def check_iomin(word: Word, word_group: Word_group):
    min_height = min(
        word.boundingbox[3] - word.boundingbox[1],
        word_group.boundingbox[3] - word_group.boundingbox[1],
    )
    intersect = min(word.boundingbox[3], word_group.boundingbox[3]) - max(
        word.boundingbox[1], word_group.boundingbox[1]
    )
    if intersect / min_height > 0.7:
        return True
    return False


def prepare_line(words):
    lines = []
    visited = [False] * len(words)
    for id_word, word in enumerate(words):
        if word.invalid_size() == 0:
            continue
        new_line = True
        for i in range(len(lines)):
            if (
                lines[i].in_same_line(word) and not visited[id_word]
            ):  # check if word is in the same line with lines[i]
                lines[i].merge_word(word)
                new_line = False
                visited[id_word] = True

        if new_line == True:
            new_line = Line()
            new_line.merge_word(word)
            lines.append(new_line)

    # print(len(lines))
    # sort line from top to bottom according top coordinate
    lines.sort(key=lambda x: x.boundingbox[1])
    return lines


def __create_word_group(word, word_group_id):
    new_word_group_ = Word_group()
    new_word_group_.list_words = list()
    new_word_group_.word_group_id = word_group_id
    new_word_group_.add_word(word)

    return new_word_group_


def __sort_line(line):
    line.list_word_groups.sort(
        key=lambda x: x.boundingbox[0]
    )  # sort word in lines from left to right

    return line


def __merge_text_for_line(line):
    line.text = ""
    for word in line.list_word_groups:
        line.text += " " + word.text

    return line


def __update_list_word_groups(line, word_group_id, word_id, line_width):

    old_list_word_group = line.list_word_groups
    list_word_groups = []

    inital_word_group = __create_word_group(old_list_word_group[0], word_group_id)
    old_list_word_group[0].word_id = word_id
    list_word_groups.append(inital_word_group)
    word_group_id += 1
    word_id += 1

    for word in old_list_word_group[1:]:
        check_word_group = True
        word.word_id = word_id
        word_id += 1

        if (
            (not list_word_groups[-1].text.endswith(":"))
            and (
                (word.boundingbox[0] - list_word_groups[-1].boundingbox[2])
                / line_width
                < MIN_WIDTH_LINE_RATIO
            )
            and check_iomin(word, list_word_groups[-1])
        ):
            list_word_groups[-1].add_word(word)
            check_word_group = False

        if check_word_group:
            new_word_group = __create_word_group(word, word_group_id)
            list_word_groups.append(new_word_group)
            word_group_id += 1
    line.list_word_groups = list_word_groups
    return line, word_group_id, word_id


def construct_word_groups_in_each_line(lines):
    line_id = 0
    word_group_id = 0
    word_id = 0
    for i in range(len(lines)):
        if len(lines[i].list_word_groups) == 0:
            continue

        # left, top ,right, bottom
        line_width = lines[i].boundingbox[2] - lines[i].boundingbox[0]  # right - left
        line_width = 30  # TODO: to remove
        lines[i] = __sort_line(lines[i])

        # update text for lines after sorting
        lines[i] = __merge_text_for_line(lines[i])

        lines[i], word_group_id, word_id = __update_list_word_groups(
            lines[i],
            word_group_id,
            word_id,
            line_width)
        lines[i].update_line_id(line_id)
        line_id += 1
    return lines


def words_to_lines(words, check_special_lines=True):  # words is list of Word instance
    # sort word by top
    words.sort(key=lambda x: (x.boundingbox[1], x.boundingbox[0]))
    # words.sort(key=lambda x: (sum(x.bbox[:])))
    number_of_word = len(words)
    # print(number_of_word)
    # sort list words to list lines, which have not contained word_group yet
    lines = prepare_line(words)

    # construct word_groups in each line
    lines = construct_word_groups_in_each_line(lines)
    return lines, number_of_word


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def merge_bboxes_to_group(bboxes_group, x_sorted_boxes):
    merged_bboxes = []
    for box_group in bboxes_group:
        merged_box = {}
        merged_box['text'] = ' '.join([x_sorted_boxes[idx]['text'] for idx in box_group])
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for idx in box_group:
            x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
            x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
            y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
            y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
        merged_box['box'] = [
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
        ]
        merged_box['list_words'] = [x_sorted_boxes[idx]['word'] for idx in box_group]
        merged_bboxes.append(merged_box)
    return merged_bboxes


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(List[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        if len(boxes) == 1:
            boxes[0]["list_words"] = [boxes[0]["word"]]
        return boxes

    # merged_groups = []
    merged_lines = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        # groups = []
        # line_idx = 0
        # groups.append([line[0]])
        # for k in range(1, len(line)):
        #     curr_box = x_sorted_boxes[line[k]]
        #     prev_box = x_sorted_boxes[line[k - 1]]
        #     dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
        #     if dist > max_x_dist:
        #         line_idx += 1
        #         groups.append([])
        #     groups[line_idx].append(line[k])

        # # Get merged boxes
        merged_line = merge_bboxes_to_group([line], x_sorted_boxes)
        merged_lines.extend(merged_line)
        # merged_group = merge_bboxes_to_group(groups,x_sorted_boxes)
        # merged_groups.extend(merged_group)

    merged_lines = sorted(merged_lines, key=lambda x: np.min(x['box'][1::2]))
    # merged_groups = sorted(merged_groups, key=lambda x: np.min(x['box'][1::2]))
    return merged_lines  # , merged_groups

# REFERENCE
# https://vigneshgig.medium.com/bounding-box-sorting-algorithm-for-text-detection-and-object-detection-from-left-to-right-and-top-cf2c523c8a85
# https://huggingface.co/spaces/tomofi/MMOCR/blame/main/mmocr/utils/box_util.py


def words_to_lines_mmocr(words: List[Word]) -> Tuple[List[Line], Optional[int]]:
    bboxes = [{"box": [w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[1], w.bbox[2], w.bbox[3], w.bbox[0], w.bbox[3]],
               "text":w.text, "word":w} for w in words]
    merged_lines = stitch_boxes_into_lines(bboxes)
    merged_groups = merged_lines  # TODO: fix code to return both word group and line
    lwords_groups = [Word_group(list_words_=merged_box["list_words"],
                                text=merged_box["text"],
                                boundingbox=[merged_box["box"][i] for i in [0, 1, 2, -1]])
                     for merged_box in merged_groups]

    llines = [Line(text=word_group.text, list_word_groups=[word_group], boundingbox=word_group.boundingbox)
              for word_group in lwords_groups]

    return llines, None  # same format with the origin words_to_lines
    # lines = [Line() for merged]


# def most_overlapping_row(rows, top, bottom, y_shift):
#     max_overlap = -1
#     max_overlap_idx = -1
#     for i, row in enumerate(rows):
#         row_top, row_bottom = row
#         overlap = min(top + y_shift, row_top) - max(bottom + y_shift, row_bottom)
#         if overlap > max_overlap:
#             max_overlap = overlap
#             max_overlap_idx = i
#     return max_overlap_idx
def most_overlapping_row(rows, row_words, top, bottom, y_shift, max_row_size, y_overlap_threshold=0.5):
    max_overlap = -1
    max_overlap_idx = -1
    overlapping_rows = []

    for i, row in enumerate(rows):
        row_top, row_bottom = row
        overlap = min(top - y_shift[i], row_top) - max(bottom - y_shift[i], row_bottom)

        if overlap > max_overlap:
            max_overlap = overlap
            max_overlap_idx = i

        if row_bottom <= top and row_top >= bottom:
            overlapping_rows.append(i)

    # Merge overlapping rows if necessary
    if len(overlapping_rows) > 1:
        merge_top = max(rows[i][0] for i in overlapping_rows)
        merge_bottom = min(rows[i][1] for i in overlapping_rows)

        if merge_top - merge_bottom <= max_row_size:
            # Merge rows
            merged_row = (merge_top, merge_bottom)
            merged_words = []
            # Remove other overlapping rows

            for row_idx in overlapping_rows[:0:-1]:  # [1,2,3] -> 3,2
                merged_words.extend(row_words[row_idx])
                del rows[row_idx]
                del row_words[row_idx]

            rows[overlapping_rows[0]] = merged_row
            row_words[overlapping_rows[0]].extend(merged_words[::-1])
            max_overlap_idx = overlapping_rows[0]

    if top - bottom - max_overlap > max_row_size * y_overlap_threshold and max_overlap < max_row_size * y_overlap_threshold:
        max_overlap_idx = -1
    return max_overlap_idx


def words_to_lines_tesseract(words: List[Word], gradient: float = 0.6) -> Tuple[List[Line], Optional[int]]:
    sorted_words = sorted(words, key=lambda x: x.bbox[0])
    rows = []
    row_words = []
    max_row_size = sorted_words[0].height
    running_y_shift = []
    for _i, word in enumerate(sorted_words):
        # if word.bbox[1] > 300 and word.bbox[3] < 500:
            # if word.text in ["Tầng", "Tel:"]:
            # print("DEBUGGING")
            
        bbox, text = word.bbox[:], word.text
        x1, y1, x2, y2 = bbox
        top, bottom = y2, y1
        max_row_size = max(max_row_size, top - bottom)
        overlap_row_idx = most_overlapping_row(rows, row_words, top, bottom, running_y_shift, max_row_size)

        if overlap_row_idx == -1:  # No overlapping row found
            new_row = (top, bottom)
            rows.append(new_row)
            row_words.append([word])
            running_y_shift.append(0)
        else:  # Overlapping row found
            row_top, row_bottom = rows[overlap_row_idx]
            new_top = max(row_top, top)
            new_bottom = min(row_bottom, bottom)
            rows[overlap_row_idx] = (new_top, new_bottom)
            row_words[overlap_row_idx].append(word)
            new_shift = bottom - row_bottom
            running_y_shift[overlap_row_idx] = gradient * running_y_shift[overlap_row_idx] + (1 - gradient) * new_shift

    # Sort rows and row_texts based on the top y-coordinate
    sorted_rows_data = sorted(zip(rows, row_words), key=lambda x: x[0][0])
    sorted_rows, sorted_row_words = zip(*sorted_rows_data)
    # return sorted_row, sorted_row_wofrds
    lwords_groups = []
    for row in sorted_row_words:
        row_text = ' '.join([word.text for word in row])
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for word in row:
            x_max = max(np.max(word.bbox[::2]), x_max)
            x_min = min(np.min(word.bbox[::2]), x_min)
            y_max = max(np.max(word.bbox[1::2]), y_max)
            y_min = min(np.min(word.bbox[1::2]), y_min)
        row_box = [x_min, y_min, x_max, y_max]

        lwords_groups.append(Word_group(list_words_=row, text=row_text, boundingbox=row_box))

    llines = [Line(text=word_group.text, list_word_groups=[word_group], boundingbox=word_group.boundingbox)
              for word_group in lwords_groups]
    return llines, None


def near(word_group1: Word_group, word_group2: Word_group):
    min_height = min(
        word_group1.boundingbox[3] - word_group1.boundingbox[1],
        word_group2.boundingbox[3] - word_group2.boundingbox[1],
    )
    overlap = min(word_group1.boundingbox[3], word_group2.boundingbox[3]) - max(
        word_group1.boundingbox[1], word_group2.boundingbox[1]
    )

    if overlap > 0:
        return True
    if abs(overlap / min_height) < 1.5:
        print("near enough", abs(overlap / min_height), overlap, min_height)
        return True
    return False


def calculate_iou_and_near(wg1: Word_group, wg2: Word_group):
    min_height = min(
        wg1.boundingbox[3] - wg1.boundingbox[1], wg2.boundingbox[3] - wg2.boundingbox[1]
    )
    overlap = min(wg1.boundingbox[3], wg2.boundingbox[3]) - max(
        wg1.boundingbox[1], wg2.boundingbox[1]
    )
    iou = overlap / min_height
    distance = min(
        abs(wg1.boundingbox[0] - wg2.boundingbox[2]),
        abs(wg1.boundingbox[2] - wg2.boundingbox[0]),
    )
    if iou > 0.7 and distance < 0.5 * (wg1.boundingboxp[2] - wg1.boundingbox[0]):
        return True
    return False


def construct_word_groups_to_kie_label(list_word_groups: list):
    kie_dict = dict()
    for wg in list_word_groups:
        if wg.kie_label == "other":
            continue
        if wg.kie_label not in kie_dict:
            kie_dict[wg.kie_label] = [wg]
        else:
            kie_dict[wg.kie_label].append(wg)

    new_dict = dict()
    for key, value in kie_dict.items():
        if len(value) == 1:
            new_dict[key] = value
            continue

        value.sort(key=lambda x: x.boundingbox[1])
        new_dict[key] = value
    return new_dict


def invoice_construct_word_groups_to_kie_label(list_word_groups: list):
    kie_dict = dict()

    for wg in list_word_groups:
        if wg.kie_label == "other":
            continue
        if wg.kie_label not in kie_dict:
            kie_dict[wg.kie_label] = [wg]
        else:
            kie_dict[wg.kie_label].append(wg)

    return kie_dict


def postprocess_total_value(kie_dict):
    if "total_in_words_value" not in kie_dict:
        return kie_dict

    for k, value in kie_dict.items():
        if k == "total_in_words_value":
            continue
        l = []
        for v in value:
            if v.boundingbox[3] <= kie_dict["total_in_words_value"][0].boundingbox[3]:
                l.append(v)

        if len(l) != 0:
            kie_dict[k] = l

    return kie_dict


def postprocess_tax_code_value(kie_dict):
    if "buyer_tax_code_value" in kie_dict or "seller_tax_code_value" not in kie_dict:
        return kie_dict

    kie_dict["buyer_tax_code_value"] = []
    for v in kie_dict["seller_tax_code_value"]:
        if "buyer_name_key" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_key"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_key"][0])
        ):
            kie_dict["buyer_tax_code_value"].append(v)
            continue

        if "buyer_name_value" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_value"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_value"][0])
        ):
            kie_dict["buyer_tax_code_value"].append(v)
            continue

        if "buyer_address_value" in kie_dict and near(
            kie_dict["buyer_address_value"][0], v
        ):
            kie_dict["buyer_tax_code_value"].append(v)
    return kie_dict


def postprocess_tax_code_key(kie_dict):
    if "buyer_tax_code_key" in kie_dict or "seller_tax_code_key" not in kie_dict:
        return kie_dict
    kie_dict["buyer_tax_code_key"] = []
    for v in kie_dict["seller_tax_code_key"]:
        if "buyer_name_key" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_key"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_key"][0])
        ):
            kie_dict["buyer_tax_code_key"].append(v)
            continue

        if "buyer_name_value" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_value"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_value"][0])
        ):
            kie_dict["buyer_tax_code_key"].append(v)
            continue

        if "buyer_address_value" in kie_dict and near(
            kie_dict["buyer_address_value"][0], v
        ):
            kie_dict["buyer_tax_code_key"].append(v)

    return kie_dict


def invoice_postprocess(kie_dict: dict):
    # all keys or values which are below total_in_words_value will be thrown away
    kie_dict = postprocess_total_value(kie_dict)
    kie_dict = postprocess_tax_code_value(kie_dict)
    kie_dict = postprocess_tax_code_key(kie_dict)
    return kie_dict


def throw_overlapping_words(list_words):
    new_list = [list_words[0]]
    for word in list_words:
        overlap = False
        area = (word.boundingbox[2] - word.boundingbox[0]) * (
            word.boundingbox[3] - word.boundingbox[1]
        )
        for word2 in new_list:
            area2 = (word2.boundingbox[2] - word2.boundingbox[0]) * (
                word2.boundingbox[3] - word2.boundingbox[1]
            )
            xmin_intersect = max(word.boundingbox[0], word2.boundingbox[0])
            xmax_intersect = min(word.boundingbox[2], word2.boundingbox[2])
            ymin_intersect = max(word.boundingbox[1], word2.boundingbox[1])
            ymax_intersect = min(word.boundingbox[3], word2.boundingbox[3])
            if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
                continue

            area_intersect = (xmax_intersect - xmin_intersect) * (
                ymax_intersect - ymin_intersect
            )
            if area_intersect / area > 0.7 or area_intersect / area2 > 0.7:
                overlap = True
        if overlap == False:
            new_list.append(word)
    return new_list


def check_iou(box1: Word, box2: Box, threshold=0.9):
    area1 = (box1.boundingbox[2] - box1.boundingbox[0]) * (
        box1.boundingbox[3] - box1.boundingbox[1]
    )
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    xmin_intersect = max(box1.boundingbox[0], box2.xmin)
    ymin_intersect = max(box1.boundingbox[1], box2.ymin)
    xmax_intersect = min(box1.boundingbox[2], box2.xmax)
    ymax_intersect = min(box1.boundingbox[3], box2.ymax)
    if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
        area_intersect = 0
    else:
        area_intersect = (xmax_intersect - xmin_intersect) * (
            ymax_intersect - ymin_intersect
        )
    union = area1 + area2 - area_intersect
    iou = area_intersect / union
    if iou > threshold:
        return True
    return False
