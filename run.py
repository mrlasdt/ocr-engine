"""
see scripts/run_ocr.sh to run     
"""
# from pathlib import Path  # add parent path to run debugger
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())


from src.utils import construct_file_path, ImageReader
from src.dto import Line
from src.ocr import OcrEngine
import argparse
import tqdm
import pandas as pd
from pathlib import Path
import json
import os
import numpy as np
from typing import Union, Tuple, List
current_dir = os.getcwd()


def get_args():
    parser = argparse.ArgumentParser()
    # parser image
    parser.add_argument("--image", type=str, required=True,
                        help="path to input image/directory/csv file")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="path to save directory")
    parser.add_argument("--include", type=str, nargs="+", default=[],
                        help="files/folders to include")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        help="files/folders to exclude")
    parser.add_argument(
        "--base_dir", type=str, required=False, default=current_dir,
        help="used when --image and --save_dir are relative paths to a base directory, default to current directory")
    parser.add_argument(
        "--export_csv", type=str, required=False, default="",
        help="used when --image is a directory. If set, a csv file contains image_path, ocr_path and label will be exported to save_dir.")
    parser.add_argument(
        "--export_img", type=bool, required=False, default=False, help="whether to save the visualize img")
    parser.add_argument("--ocr_kwargs", type=str, required=False, default="")
    opt = parser.parse_args()
    return opt


def load_engine(opt) -> OcrEngine:
    print("[INFO] Loading engine...")
    kw = json.loads(opt.ocr_kwargs) if opt.ocr_kwargs else {}
    engine = OcrEngine(**kw)
    print("[INFO] Engine loaded")
    return engine


def convert_relative_path_to_positive_path(tgt_dir: Path, base_dir: Path) -> Path:
    return tgt_dir if tgt_dir.is_absolute() else base_dir.joinpath(tgt_dir)


def get_paths_from_opt(opt) -> Tuple[Path, Path]:
    # BC\ kiem\ tra\ y\ te -> BC kiem tra y te
    img_path = opt.image.replace("\\ ", " ").strip()
    save_dir = opt.save_dir.replace("\\ ", " ").strip()
    base_dir = opt.base_dir.replace("\\ ", " ").strip()
    input_image = convert_relative_path_to_positive_path(
        Path(img_path), Path(base_dir))
    save_dir = convert_relative_path_to_positive_path(
        Path(save_dir), Path(base_dir))
    if not save_dir.exists():
        save_dir.mkdir()
        print("[INFO]: Creating folder ", save_dir)
    return input_image, save_dir


def process_img(img: Union[str, np.ndarray], save_dir_or_path: str, engine: OcrEngine, export_img: bool) -> None:
    save_dir_or_path = Path(save_dir_or_path)
    if isinstance(img, np.ndarray):
        if save_dir_or_path.is_dir():
            raise ValueError(
                "numpy array input require a save path, not a save dir")
    page = engine(img)
    save_path = str(save_dir_or_path.joinpath(Path(img).stem + ".txt")
                    ) if save_dir_or_path.is_dir() else str(save_dir_or_path)
    page.write_to_file('word', save_path)
    if export_img:
        page.save_img(save_path.replace(".txt", ".jpg"), is_vnese=True, )


def process_dir(
        dir_path: str, save_dir: str, engine: OcrEngine, export_img: bool, lexcludes: List[str] = [], lincludes: List[str] = [],
        ddata: dict = {"img_path": list(),
                       "ocr_path": list(),
                       "label": list()}) -> None:
    dir_path = Path(dir_path)
    # save_dir_sub = Path(construct_file_path(save_dir, dir_path, ext=""))
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    for img_path in (pbar := tqdm.tqdm(dir_path.iterdir())):
        pbar.set_description(f"Processing {dir_path}")
        if (lincludes and img_path.name not in lincludes) or (img_path.name in lexcludes):
            continue #only process desired files/foders
        if img_path.is_dir():
            save_dir_sub = save_dir.joinpath(img_path.stem)
            process_dir(img_path, str(save_dir_sub), engine, ddata)
        elif img_path.suffix.lower() in ImageReader.supported_ext:
            simg_path = str(img_path)
            # try:
            img = ImageReader.read(
                simg_path) if img_path.suffix != ".pdf" else ImageReader.read(simg_path)[0]
            save_path = str(Path(save_dir).joinpath(
                img_path.stem + ".txt"))
            process_img(img, save_path, engine, export_img)
            # except Exception as e:
            #     print('[ERROR]: ', e, ' at ', simg_path)
            #     continue
            ddata["img_path"].append(simg_path)
            ddata["ocr_path"].append(save_path)
            ddata["label"].append(dir_path.stem)
            # ddata.update({"img_path": img_path, "save_path": save_path, "label": dir_path.stem})
    return ddata


def process_csv(csv_path: str, engine: OcrEngine) -> None:
    df = pd.read_csv(csv_path)
    if not 'image_path' in df.columns or not 'ocr_path' in df.columns:
        raise AssertionError('Cannot fing image_path in df headers')
    for row in df.iterrows():
        process_img(row.image_path, row.ocr_path, engine)


if __name__ == "__main__":
    opt = get_args()
    engine = load_engine(opt)
    print("[INFO]: OCR engine settings:", engine.settings)
    img, save_dir = get_paths_from_opt(opt)

    lskip_dir = []
    if img.is_dir():
        ddata = process_dir(img, save_dir, engine, opt.export_img, opt.exclude, opt.include)
        if opt.export_csv:
            pd.DataFrame.from_dict(ddata).to_csv(
                Path(save_dir).joinpath(opt.export_csv))
    elif img.suffix in ImageReader.supported_ext:
        process_img(str(img), save_dir, engine, opt.export_img)
    elif img.suffix == '.csv':
        print("[WARNING]: Running with csv file will ignore the save_dir argument. Instead, the ocr_path in the csv would be used")
        process_csv(img, engine)
    else:
        raise NotImplementedError('[ERROR]: Unsupported file {}'.format(img))
