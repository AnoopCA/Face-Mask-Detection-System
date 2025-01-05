import numpy as np
import pandas as pd
import os
import sys
import torch
import cv2
from torchvision import transforms
from PIL import Image

#from train import FaceMaskDetection

sys.path.append(os.path.abspath(r"D:\ML_Projects\Face-Mask-Detection-System\YOLOv3"))
from model import YOLOv3
import config
from utils import cells_to_bboxes, non_max_suppression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.04

#img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\validation_images'
img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images'
#model_path = r'D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_12_e1024.pth'
#model_path = r'D:\ML_Projects\Face-Mask-Detection-System\References\YOLOv3\Models\fmd_yolov3_2.pth.tar'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_10.pth.tar'

#model = FaceMaskDetection()
model = YOLOv3(num_classes=config.NUM_CLASSES)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

#model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
                                 transforms.Resize((416, 416)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

#df = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    original_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    #results = model(img).to(device)
    #results = [res.to(device) for res in model(img)]
    results = get_bboxes(img, model, config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD,)
    for result in results:
        for r in result:
            #x1, y1, x2, y2, score = r
            print(f"model output: {r}")
            score = 0
            if score > 0.001:
                x1 = int(x1 * original_img.shape[1])
                y1 = int(y1 * original_img.shape[0])
                x2 = int(x2 * original_img.shape[1])
                y2 = int(y2 * original_img.shape[0])
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #cv2.imshow("Prediction", original_img)
    #cv2.waitKey(2000)

    #output_img_path = os.path.join(img_out, f"output_{img_name}")
    #cv2.imwrite(output_img_path, original_img)


def get_bboxes(
    x,
    model,
    iou_threshold,
    anchors,
    threshold,
):
    # make sure model is in eval before get bboxes
    model.eval()
    all_pred_boxes = []
    with torch.no_grad():
        predictions = model(x)
    bboxes = []
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    nms_boxes = non_max_suppression(
        bboxes,
        iou_threshold=iou_threshold,
        threshold=threshold
    )

    model.train()
    return nms_boxes

