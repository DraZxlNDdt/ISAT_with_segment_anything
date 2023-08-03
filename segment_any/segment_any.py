# -*- coding: utf-8 -*-
# @Author  : LG

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
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
        self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.image = None

    def switch_to_cpu(self):
        self.sam.to('cpu')
        torch.cuda.empty_cache()

    def switch_to_device(self):
        self.sam.to(self.device)

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
        print('generated...')
        for mask in masks:
            mask["segmentation"] = torch.tensor(mask["segmentation"]).to(device)
            self.add_RGB_segment(result, mask["segmentation"], annotation, colors)

    def run(self, image_root, image_name):
        image_path = os.path.join(image_root, image_name)
        seg_path = os.path.join(image_root, 'bss_'+image_name)
        if not os.path.exists(seg_path):
            print('gg')
            return
        
        mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=64, points_per_batch=32)
        device = self.device
    
        colors_pattern = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]
        colors = torch.tensor(colors_pattern, dtype=torch.uint8).to(device)
        
        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)

        annotation = Image.open(seg_path).convert('RGB')
        annotation = np.array(annotation)
        annotation = torch.tensor(annotation).to(device)
        result =  torch.full(annotation.shape, 255, dtype=torch.uint8).to(device)
        
        step = 1024
        for h_start in range(0, image.shape[0], step):
            for w_start in range(0, image.shape[1], step):
                h_end = np.min([h_start+step, image.shape[0]])
                w_end = np.min([w_start+step, image.shape[1]])
                print(h_start, h_end, w_start, w_end, device)
                self.update_patch(mask_generator, image[h_start:h_end, w_start:w_end], 
                            annotation[h_start:h_end, w_start:w_end], result[h_start:h_end, w_start:w_end], device, colors)
                
        save_path = os.path.join(image_root, 'bss_'+image_name)
        Image.fromarray(result.cpu().numpy()).save(save_path)
        return save_path