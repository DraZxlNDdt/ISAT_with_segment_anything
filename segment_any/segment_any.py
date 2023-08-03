# -*- coding: utf-8 -*-
# @Author  : LG

from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from PIL import Image
import os


class SegAny:
    def __init__(self, checkpoint):
        if 'vit_b' in checkpoint:
            self.model_type = "vit_b"
        elif 'vit_l' in checkpoint:
            self.model_type = "vit_l"
        elif 'vit_h' in checkpoint:
            self.model_type = "vit_h"
        else:
            raise ValueError('The checkpoint named {} is not supported.'.format(checkpoint))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.image = None

    def switch_to_cpu(self):
        self.predictor.model = self.predictor.model.to('cpu')
        torch.cuda.empty_cache()

    def switch_to_device(self):
        self.predictor.model = self.predictor.model.to(self.device)

    def set_image(self, image):
        self.image = image
        self.predictor.set_image(image)

    def reset_image(self):
        self.predictor.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict(self, input_point, input_label):
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        for _ in range(20):
            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                multimask_output=True,
            )
        torch.cuda.empty_cache()
        return masks[np.argmax(scores)]

    def add_RGB_segment(self, result, segmentation_mask, annotation, colors):
        stats = [(annotation[segmentation_mask] == color).all(dim=1).sum().item() for color in colors]
        index = np.argmax(stats)
        if (index != 0):
            result[segmentation_mask] = colors[index]
    
    def update_patch(self, mask_generator, image, annotation, result, device, colors):
        masks = mask_generator.generate(image)

        for mask in masks:
            mask["segmentation"] = torch.tensor(mask["segmentation"]).to(device)
            self.add_RGB_segment(result, mask["segmentation"], annotation, colors)

    def run(self, image_root, image_name):
        image_path = os.path.join(image_root, image_name)
        seg_path = os.path.join(image_root, 'bss_'+image_name)
        if not os.path.exists(seg_path):
            print('gg')
            return
        
        return
        mask_generator = None
        COLOR_MAP = dict({
            'background': (0, 0, 0),
            'ship': (0, 0, 63),
            'storage_tank': (0, 191, 127),
            'baseball_diamond': (0, 63, 0),
            'tennis_court': (0, 63, 127),
            'basketball_court': (0, 63, 191),
            'ground_Track_Field': (0, 63, 255),
            'bridge': (0, 127, 63),
            'large_Vehicle': (0, 127, 127),
            'small_Vehicle': (0, 0, 127),
            'helicopter': (0, 0, 191),
            'swimming_pool': (0, 0, 255),
            'roundabout': (0, 63, 63),
            'soccer_ball_field': (0, 127, 191),
            'plane': (0, 127, 255),
            'harbor': (0, 100, 155),
        })

        colors = torch.tensor(list(COLOR_MAP.values()), dtype=torch.uint8).to(device)
        

        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)

        file_name = os.path.basename(image_path)

        annotation = Image.open(os.path.join(segpath, file_name)).convert('RGB')
        annotation = np.array(annotation)
        annotation = torch.tensor(annotation).to(device)
        result = torch.zeros_like(annotation).to(device)
        
        step = 1024
        for h_start in range(0, image.shape[0], step):
            for w_start in range(0, image.shape[1], step):
                h_end = np.min([h_start+step, image.shape[0]])
                w_end = np.min([w_start+step, image.shape[1]])
                # print(h_start, h_end, w_start, w_end)
                self.update_patch(mask_generator, image[h_start:h_end, w_start:w_end], 
                            annotation[h_start:h_end, w_start:w_end], result[h_start:h_end, w_start:w_end], device, colors)
                
        save_path = segpath+'_fixed3_step='+str(step)+'_background=black_pps=64'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        Image.fromarray(result.cpu().numpy()).save(os.path.join(save_path, file_name))
