## Introduction
This repo serve as source code storage for the Standalone SATRN Text Recognizer packages.  
Installing this package requires 3 additional packages: PyTorch, MMCV, and colorama.


## Installation
```shell
conda create -n sdsvtr-env python=3.8
conda activate sdsvtr-env
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
mim install mmcv-full
pip install colorama
git clone https://github.com/moewiee/sdsvtr.git
cd sdsvtr
pip install -v -e .
```

## Basic Usage
```python
from sdsvtr import StandaloneSATRNRunner
runner = StandaloneSATRNRunner(version='satrn-lite-general-pretrain-20230106', return_confident=False, use_cuda=False)
```  

The `version` parameter accepts version names declared in `sdsvtr.factory.online_model_factory` or a local path such as `$DIR\model.pth`. To check for available versions in the hub, run:
```python
import sdsvtr
print(sdsvtr.__hub_available_versions__)
```  

Naturally, a `StandaloneSATRNRunner` instance assumes the input to be one of the following: an instance of `np.ndarray`, an instance of `str`, a list of `np.ndarray`, or a list of `str`, for examples:
```python
import numpy as np
from sdsvtr import StandaloneSATRNRunner
runner = StandaloneSATRNRunner(version='satrn-lite-general-pretrain-20230106', return_confident=False, use_cuda=False)

dummy_list = [np.ndarray((32,128,3)) for _ in range(100)]
result = runner(dummy_list)
```  

To run with a specific batchsize, try:
```python
import numpy as np
from sdsvtr import StandaloneSATRNRunner
runner = StandaloneSATRNRunner(version='satrn-lite-general-pretrain-20230106', return_confident=False, device='cuda:0')

dummy_list = [np.ndarray(1,3,32,128) for _ in range(100)]
bs = min(32, len(imageFiles)) # batchsize = 32

all_results = []
while len(dummy_list) > 0:
    dummy_batch = dummy_list[:bs]
    dummy_list = dummy_list[bs:]
    all_results += runner(dummy_batch)
```

## Version Changelog
* **[0.0.1]**  
Initial version with specified features.  
  

* **[0.0.2]**  
Update online model hub.  
  

* **[0.0.3]**  
Update API now able to inference with 4 types of inputs: list/instance of `np.ndarray`/`str`  
Update API interface with `return_confident` parameter.  
Update `wget` check and `sha256` check for model hub retrieval.  

* **[0.0.4]**  
Update decoder module with EarlyStopping mechanism to possibly improve inference speed on short sequences.  
Update API interface with optional argument `max_seq_len_overwrite` to overwrite checkpoint's `max_seq_len` config.

* **[0.0.5]**  
Allow inference on a specific device
