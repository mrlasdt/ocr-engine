import mmcv
import numpy as np
import cv2
import torch


class DetectorDataPipeline:
    
    def __init__(self, 
                 img_scale,
                 device):
        self.scale = img_scale
        self.device = device
        
    def load(self, img):
        if isinstance(img, str):
            return cv2.imread(img)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise ValueError(f'img input must be a str/np.ndarray, got {type(img)}')
    
    def resize(self, img):
        origin_shape = img.shape[:2]
        img = mmcv.imrescale(img,
                             self.scale,
                             return_scale=False,
                             interpolation='bilinear',
                             backend='cv2')
            
        return img, origin_shape, np.array(self.scale)
            
    def pad(self, img):
        if self.scale is not None:
            width = max(self.scale[1] - img.shape[1], 0)
            height = max(self.scale[0] - img.shape[0], 0)
            padding = (0, 0, width, height)
        
        img = cv2.copyMakeBorder(img,
                                 padding[1],
                                 padding[3],
                                 padding[0],
                                 padding[2],
                                 cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114,))
        
        return img
        
    def to_tensor(self, img):
        
        img = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0)
        img = img.to(device=self.device)
        
        return img
        
    def __call__(self, img):
        img = self.load(img)
        img, origin_shape, new_shape = self.resize(img)
        img = self.pad(img)
        img = self.to_tensor(img)
        
        return img, origin_shape, new_shape
    

class AutoRotateDetectorDataPipeline(DetectorDataPipeline):

    def __call__(self, img):
        img = self.load(img)
        imgs = []
        imgs_np = []

        for flag in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            img_t = img if flag is None else cv2.rotate(img, flag)
            imgs_np.append(img_t)
            img_t, _, _ = self.resize(img_t)
            img_t = self.pad(img_t)
            img_t = self.to_tensor(img_t)

            imgs.append(img_t)

        return imgs, imgs_np