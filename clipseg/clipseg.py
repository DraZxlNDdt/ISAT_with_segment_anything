# -*- coding: utf-8 -*-
# @Author  : zengxl

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from PIL import Image
import os


class CLIPSEG:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.image = None
        colors_pattern = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
        self.color = torch.tensor(colors_pattern, dtype=torch.uint8).to(self.device)
        self.prompts = list(('building', 'road', 'water', 'barren', 'forest', 'agricultural'))
        self.prompts = list(('building', 'tree', 'flower', 'road', 'sky', 'agricultural'))
        self.class_num = len(self.prompts)
        threshold = 0.3

    def set_image(self, image):
        self.image = image

    def reset_image(self):
        self.image = None
        torch.cuda.empty_cache()

    def save(self, image_np, final_pred, file_path, threshold = 0.3):
        image_ori = Image.fromarray(image_np)
        prob, pred = final_pred.max(dim=0)
        pred += 1
        # 置信度低的标签为背景

        pred[prob<threshold] = 0
        ans_map = self.color[pred]
        img = Image.fromarray(ans_map.cpu().numpy(), mode='RGB')
        img.save(file_path)
        return img, file_path


    def get_pred(self, image_np, debug=0):
        step = 352
        image_ori = Image.fromarray(image_np)
        final_shape = np.ceil(np.array(image_np.shape)[0:2]/step).astype(np.int32)*step
        final_shape = (self.class_num, final_shape[0], final_shape[1])
        final_pred = torch.zeros(final_shape).to(self.device)
        # maybe use unfold``
        for w_start in range(0, final_shape[2], step):
            for h_start in range(0, final_shape[1], step):
                image = image_ori.crop((w_start, h_start, step+w_start, step+h_start))
                inputs = self.processor(text=self.prompts, images=[image] * self.class_num, padding="max_length", return_tensors="pt")
                inputs = inputs.to(self.device)
                # predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # visualize prediction
                preds = outputs.logits
                final_pred[:, h_start:h_start+step, w_start:w_start+step] = preds

        final_pred = final_pred[:, :image_ori.height, :image_ori.width]
        return final_pred.sigmoid()

    def predict(self, root, filename):
        self.model = self.model.to(self.device)
        img = self.image
        pred = self.get_pred(img)
        final_img, clipseg_path = self.save(img, pred, os.path.join(root, 'bss_'+filename))
        img = None
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        return final_img, clipseg_path