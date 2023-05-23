## Introduction
This repo serve as source code storage for the Standalone YoloX Detection packages.  
Installing this package requires 2 additional packages: PyTorch and MMCV.


## Installation
```shell
conda create -n sdsvtd-env python=3.8
conda activate sdsvtd-env
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
pip install mmcv-full
git clone https://github.com/moewiee/sdsvtd.git
cd sdsvtd
pip install -v -e .
```

## Basic Usage
```python
from sdsvtd import StandaloneYOLOXRunner
runner = StandaloneYOLOXRunner(version='yolox-s-general-text-pretrain-20221226', device='cpu')
```  

The `version` parameter accepts version names declared in `sdsvtd.factory.online_model_factory` or a local path such as `$DIR\model.pth`. To check for available versions in the hub, run:
```python
import sdsvtd
print(sdsvtd.__hub_available_versions__)
```  

Naturally, a `StandaloneYOLOXRunner` instance assumes the input to be an instance of `np.ndarray` or an instance of `str` (batch inferece is not supported), for examples:
```python
import numpy as np
from sdsvtd import StandaloneYOLOXRunner
runner = StandaloneYOLOXRunner(version='yolox-s-general-text-pretrain-20221226', device='cpu')

dummy_input = np.ndarray(500, 500, 3)
result = runner(dummy_input)
```  

**Note:** Output of `StandaloneYOLOXRunner` instance will be in format `List[np.ndarray]` with each list element corresponds to one class. Each `np.ndarray` will be a 5-d vector `[x1 y1 x2 y2 confident]` with coordinates rescaled to fit the original image size.  

**AutoRotation:**  
From version 0.1.0, `sdsvtd` support *AutoRotation* feature which accept a `np.ndarray/str` as input and return an straight rotated image (only available rotation degrees are 90, 180 and 270) of type `np.ndarray` and its bounding boxes of type `List[np.ndarray]`. Usage:
```python
import numpy as np
from sdsvtd import StandaloneYOLOXRunner
runner = StandaloneYOLOXRunner(version='yolox-s-general-text-pretrain-20221226', device='cpu', auto_rotate=True)

rotated_image, result = runner(cv2.imread('path-to-image')) # or
rotated_image, result = runner(np.ndarray)
```

## Version Changelog
* **[0.0.1]**  
Initial version with specified features.  

* **[0.0.2]**  
Update more versions in model hub.  

* **[0.0.3]**  
Update feature to specify running device while initialize runner. [Issue](https://github.com/moewiee/sdsvtd/issues/2)  

* **[0.0.4]**  
Fix a minor bugs when existing hub/local version != current hub/local version. 

* **[0.0.5]**  
Update model hub with handwritten text line detection version. 

* **[0.1.0]**  
Update new feature: Auto Rotation.  

* **[0.1.1]**  
Fix a bug in API inference class where return double result with auto_rotate=False.  

* **[0.1.2]**
Fix a bug in rotate feature where rotator_version was ignored.  

