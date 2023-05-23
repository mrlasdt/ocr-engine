from .model import SingleStageDetector, AutoRotateDetector
import numpy as np

class StandaloneYOLOXRunner:
    
    def __init__(self,
                 version, 
                 device,
                 auto_rotate=False,
                 rotator_version=None):
        self.model = SingleStageDetector(version,
                                         device)
        self.auto_rotate = auto_rotate
        if self.auto_rotate:
            if rotator_version is None:
                rotator_version = version
            self.rotator = AutoRotateDetector(rotator_version,
                                              device)
        
        self.warmup_()
        
    def warmup_(self):
        ''' Call on dummpy input to warm up instance '''
        x = np.ndarray((1280, 1280, 3)).astype(np.uint8)
        self.model(x)
        if self.auto_rotate:
            self.rotator(x)

        
    def __call__(self, img):
        if self.auto_rotate:
            img = self.rotator(img)
        result = self.model(img)
        
        return result if not self.auto_rotate else (img, result)