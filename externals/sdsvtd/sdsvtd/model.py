import torch
import torch.nn as nn
import numpy as np
from .backbone import CSPDarknet
from .neck import YOLOXPAFPN
from .bbox_head import YOLOXHead
from .transform import DetectorDataPipeline, AutoRotateDetectorDataPipeline
from .factory import _get as get_version


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]
    
    
def normalize_bbox(bboxes, scale):
    for i in range(len(bboxes)):
        bboxes[i][:,:4] /= scale
    return bboxes


class SingleStageDetector(nn.Module):
    
    def __init__(self,
                 version,
                 device):
        super(SingleStageDetector, self).__init__()
        
        assert 'cpu' in device or 'cuda' in device
        
        checkpoint = get_version(version)
        pt = torch.load(checkpoint, 'cpu')
        
        self.pipeline = DetectorDataPipeline(**pt['pipeline_args'], device=device)
        self.backbone = CSPDarknet(**pt['backbone_args'])
        self.neck = YOLOXPAFPN(**pt['neck_args'])
        self.bbox_head = YOLOXHead(**pt['bbox_head_args'])
        self.load_state_dict(pt['state_dict'], strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
        self = self.to(device=device)
            
        print(f'Text detection load from version {version}.')

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, img):
        """Test function without test-time augmentation.

        Args:
            img (np.ndarray): Images with shape (H, W, C) or
            img (str): Path to image.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
            The list corresponds to each class.
        """
        img, origin_shape, new_shape = self.pipeline(img)
        scale = min(new_shape / origin_shape)
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test_bboxes(feat)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ][0]
        bbox_results = normalize_bbox(bbox_results, scale)
        return bbox_results
    

class AutoRotateDetector(nn.Module):
    def __init__(self,
                 version,
                 device):
        super(AutoRotateDetector, self).__init__()
        
        assert 'cpu' in device or 'cuda' in device
        
        checkpoint = get_version(version)
        pt = torch.load(checkpoint, 'cpu')
        
        self.pipeline = AutoRotateDetectorDataPipeline(**pt['pipeline_args'], device=device)
        self.backbone = CSPDarknet(**pt['backbone_args'])
        self.neck = YOLOXPAFPN(**pt['neck_args'])
        self.bbox_head = YOLOXHead(**pt['bbox_head_args'], nms_score_thr=0.8)
        self.load_state_dict(pt['state_dict'], strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
        self = self.to(device=device)
            
        print(f'Auto rotate detector load from version {version}.')

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, img):
        """Test function without test-time augmentation.

        Args:
            img (np.ndarray): Images with shape (H, W, C) or
            img (str): Path to image.

        Returns:
            np.ndarray: Straight rotated image.
        """
        imgs, imgs_np = self.pipeline(img)
        maxCount = -1
        maxCountRot = None
        for idx, img in enumerate(imgs):
            currentCount = 0
            feat = self.extract_feat(img)
            results_list = self.bbox_head.simple_test_bboxes(feat)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ][0]
            for class_result in bbox_results:
                currentCount += len(class_result)
            if currentCount > maxCount:
                maxCount = currentCount
                maxCountRot = idx
        return imgs_np[maxCountRot]