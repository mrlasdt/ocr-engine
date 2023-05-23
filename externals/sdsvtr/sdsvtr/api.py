
import torch
import torch.nn as nn

from colorama import Fore, Style

from .converter import AttnConvertor
from .backbone import ResNetABI
from .encoder import SatrnEncoder
from .decoder import NRTRDecoder
from .transform import DataPipelineSATRN
from .fp16_utils import patch_norm_fp32
from .factory import _get as get_version


class SATRN(nn.Module):
    """Standalone implementation for SATRN encode-decode recognizer."""

    def __init__(self,
                 version,
                 return_confident=False, 
                 device='cpu',
                 max_seq_len_overwrite=None):
        
        super().__init__()
        
        checkpoint = get_version(version)
        
        pt = torch.load(checkpoint, 'cpu')
        if device == 'cpu':
            print(Fore.RED + 'Warning: You are using CPU inference method. Init with device=cuda:<id> to run with CUDA method.' + Style.RESET_ALL)
        
        self.pipeline = DataPipelineSATRN(**pt['pipeline_args'], device=device)

        # Convertor
        self.label_convertor = AttnConvertor(**pt['label_convertor_args'], return_confident=return_confident)

        # Backbone
        self.backbone = ResNetABI(**pt['backbone_args'])

        # Encoder module
        self.encoder = SatrnEncoder(**pt['encoder_args'])

        # Decoder module
        decoder_max_seq_len = max_seq_len_overwrite if max_seq_len_overwrite is not None else pt['max_seq_len']
        self.decoder = NRTRDecoder(
            **pt['decoder_args'],
            max_seq_len=decoder_max_seq_len,
            num_classes=self.label_convertor.num_classes(),
            start_idx=self.label_convertor.start_idx,
            padding_idx=self.label_convertor.padding_idx,
            return_confident=return_confident,
            end_idx=self.label_convertor.end_idx
        )
                
        self.load_state_dict(pt['state_dict'], strict=True)
        print(f'Text recognition from version {version}.')
        
        if device != 'cpu':
            self = self.to(device)
            self = self.half()
            patch_norm_fp32(self)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def extract_feat(self, img):
        x = self.backbone(img)

        return x

    def forward(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.

        Returns:
            list[str]: Text label result of each image.
        """
        img = self.pipeline(img)
        feat = self.extract_feat(img)
        out_enc = self.encoder(feat)
        out_dec = self.decoder(out_enc).cpu().numpy()
        label_strings = self.label_convertor(out_dec)

        return label_strings
    

class StandaloneSATRNRunner:
    def __init__(self,
                 version,
                 return_confident,
                 device='cpu',
                 max_seq_len_overwrite=None):
        self.device = device
        self.model = SATRN(version=version,
                           return_confident=return_confident,
                           device=self.device,
                           max_seq_len_overwrite=max_seq_len_overwrite)
    
    def __call__(self, imgs):
        results = self.model(imgs)
        
        return results