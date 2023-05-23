import numpy as np
from sdsvtr import StandaloneSATRNRunner
runner = StandaloneSATRNRunner(version='satrn-lite-general-pretrain-20230106', return_confident=False, device='cuda:1')

dummy_list = [np.ndarray((32,128,3)) for _ in range(100)]
bs = 32

all_results = []
while len(dummy_list) > 0:
    dummy_batch = dummy_list[:bs]
    dummy_list = dummy_list[bs:]
    all_results += runner(dummy_batch)