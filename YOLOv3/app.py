import numpy as np
import pandas as pd
import os
import sys
import torch
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(r"D:\ML_Projects\Face-Mask-Detection-System\YOLOv3"))
from model import YOLOv3
import config
from utils import cells_to_bboxes, non_max_suppression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_11.pth.tar'

def get_bboxes(x, model, iou_threshold, anchors, threshold):
    model.eval()
    all_pred_boxes = []
    with torch.no_grad():
        predictions = model(x)
    bboxes = [[] for _ in range(1)]
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    nms_boxes = non_max_suppression(bboxes[0], iou_threshold=iou_threshold, threshold=threshold)
    model.train()
    return nms_boxes

model = YOLOv3(num_classes=config.NUM_CLASSES)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
                                 transforms.Resize((224, 224)), #(416, 416)
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 10))

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    original_img = img
    original_img_np = np.array(img)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    results = get_bboxes(img, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
    for r in results:
        class_pred, prob_score, x1, y1, x2, y2 = r
        if prob_score > 0.001:
            width, height = original_img.size
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            cv2.rectangle(original_img_np, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # Update the figure with the new image
    ax.clear()  # Clear the previous image
    ax.imshow(original_img_np)
    ax.axis("off")
    ax.set_title(f"Prediction: {img_name}")
    plt.draw()  # Redraw the updated image
    plt.pause(4)  # Pause to simulate the video effect, adjust as necessary

plt.ioff()  # Turn off interactive mode to stop dynamic updates
    #break
    #output_img_path = os.path.join(img_out, f"output_{img_name}")
    #cv2.imwrite(output_img_path, original_img)
