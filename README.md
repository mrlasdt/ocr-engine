# OCR Engine

OCR Engine is a Python package that combines text detection and recognition models from [mmdet](https://github.com/open-mmlab/mmdetection) and [mmocr](https://github.com/open-mmlab/mmocr) to perform Optical Character Recognition (OCR) on various inputs. The package currently supports three types of input: a single image, a recursive directory, or a csv file.

## Installation

To install OCR Engine, clone the repository and install the required packages:

```bash
git clone git@github.com:mrlasdt/ocr-engine.git
cd ocr-engine
pip install -r requirements.txt

```


## Usage

To use OCR Engine, simply run the `ocr_engine.py` script with the desired input type and input path. For example, to perform OCR on a single image:

```css
python ocr_engine.py --input_type image --input_path /path/to/image.jpg
```

To perform OCR on a recursive directory:

```css
python ocr_engine.py --input_type directory --input_path /path/to/directory/

```

To perform OCR on a csv file:


```
python ocr_engine.py --input_type csv --input_path /path/to/file.csv
```

OCR Engine will automatically detect and recognize text in the input and output the results in a CSV file named `ocr_results.csv`.

## Contributing

If you would like to contribute to OCR Engine, please fork the repository and submit a pull request. We welcome contributions of all types, including bug fixes, new features, and documentation improvements.

## License

OCR Engine is released under the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for more information.
