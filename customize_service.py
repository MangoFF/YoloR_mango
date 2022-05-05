# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
"""
from model_service.pytorch_model_service import PTServingBaseService
import argparse
from PIL import Image
import numpy as np
from loguru import logger
import cv2
import os
import torch

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]



class PTVisionService(PTServingBaseService):
#class PTVisionService:
    def __init__(self, model_name,model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name,model_path)

        # 加载标签
        self.label = [0,1,2,3,4,5,6,7,8,9]
        self.img_size = 1024

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # 调用自定义函数加载模型
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.half()
        # Configure
        self.model.eval()
        self.input_image_key = 'images'
        self.data = {}
        self.data['nc'] = 10
        self.data['names'] = ['lighthouse', 'sailboat', 'buoy', 'railbar', 'cargoship', 'navalvessels', 'passengership', 'dock', 'submarine', 'fishingboat']
        self.class_map = self.data['names']
        self.conf_thres = 0.01
        self.iou_thres = 0.65
        self.img_size = 640
    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content).convert('RGB')
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                preprocessed_data[k] = [image, file_name]
        # print(preprocessed_data)
        return preprocessed_data

    def _postprocess(self, data):

        return data

    def _inference(self, data):
        result = {}
        img_org = data['images'][0]

        # Padded resize
        img = letterbox(img_org, new_shape=self.img_size, auto_size=64)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if True else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_org.shape).round()
        # Apply NMS

        bboxes = det[:, 0:4]
        result['detection_classes'] = []
        result['detection_scores'] = []
        result['detection_boxes'] = []

        for p, b in zip(det.tolist(), bboxes.tolist()):
            b = [b[1], b[0], b[3], b[2]]  # y1 x1 y2 x2
            result['detection_classes'].append(self.class_map[int(p[5])])

            result['detection_scores'].append(round(p[4], 5))

            result['detection_boxes'].append([round(x, 3) for x in b])
        return result

if __name__=="__main__":
    data={}
    img= Image.open("yolor/inference/images/boat.jpeg")
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    data["images"]=[img]
    data["images"].append("mango")
    p=PTVisionService(model_name="yolox",model_path="yolor/best.pt")
    print(p._inference(data))
