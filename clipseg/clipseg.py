# -*- coding: utf-8 -*-
# @Author  : zengxl

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from PIL import Image

from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex
import glob
import os


class CLIPSEG:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.image = None
        colors_pattern = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
        self.color = torch.tensor(colors_pattern, dtype=torch.uint8).to(self.device)
        self.prompts = list(('building', 'road', 'water', 'barren', 'forest', 'agricultural'))
        self.class_num = len(self.prompts)
        threshold = 0.3

    def set_image(self, image):
        self.image = image

    def reset_image(self):
        self.image = None
        torch.cuda.empty_cache()

    def save(self, image_np, final_pred, file_name, threshold_list = [0.3]):
        image_ori = Image.fromarray(image_np)
        prob, pred = final_pred.max(dim=0)
        pred += 1
        # 置信度低的标签为背景

        for threshold in threshold_list:
            # ans_map[prob<threshold] = 255
            # img = Image.fromarray(ans_map.cpu().numpy(), mode='RGB')
            # display(Image.blend(image_ori, img, 0.6))
            pred[prob<threshold] = 0
            ans_map = self.color[pred]
            img = Image.fromarray(ans_map.cpu().numpy(), mode='RGB')
            img.save('Result/'+file_name)

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

    def predict(self):
        threshold_list = [0.3, 0.4, 0.5]
        img = self.image
        pred = self.get_pred(img)
        self.save(img, pred, 'bb.jpg', threshold_list)
        prob, y_pred = pred.max(dim=0)
        y_pred += 1
        img = None
        torch.cuda.empty_cache()

if __name__ == "__main__":
    cs = CLIPSEG()
    cs.predict()