# -*- coding: utf-8 -*-
# @Author  : zengxl

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from PIL import Image
import os
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import numpy as np

class RVSA:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_file = 'rvsa/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py'
        checkpoint_file = 'rvsa/checkpoints/vitae_rvsa_kvdiff.pth'
        self.model = init_segmentor(config_file, checkpoint_file, device='cpu')
        self.image = None

    def set_image(self, image):
        self.image = image

    def reset_image(self):
        self.image = None
        torch.cuda.empty_cache()

    def predict(self, root, filename):
        self.model = self.model.to(self.device)
        img = self.image
        result = inference_segmentor(self.model, img)
        path = os.path.join(root, 'bss_'+filename)
        self.model.show_result(img, result, out_file=path, opacity=1)
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        return result, path
    