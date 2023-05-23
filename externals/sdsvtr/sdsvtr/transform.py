import torchvision.transforms.functional as TF
import cv2
import torch

class DataPipelineSATRN:
    def __init__(self,
                 resize_height,
                 resize_width,
                 norm_mean,
                 norm_std,
                 device='cpu'):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.device = device

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        datas = []
        for img in imgs:
            if isinstance(img, str):
                img = cv2.imread(img)
            data = torch.from_numpy(cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR))
            datas.append(data)
        
        data = torch.stack(datas, dim=0)
        data = data.to(self.device)
        data = data.float().div_(255.).permute((0,3,1,2))
        TF.normalize(data, mean=self.norm_mean, std=self.norm_std, inplace=True)
        
        return data.half() if self.device != 'cpu' else data