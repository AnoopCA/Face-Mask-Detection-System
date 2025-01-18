import numpy as np
import pandas as pd
import os
import sys
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(r"D:\ML_Projects\Face-Mask-Detection-System\YOLOv3"))
from model import YOLOv3
import config
from utils import cells_to_bboxes, non_max_suppression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_12.pth.tar'

def get_bboxes(x, model, iou_threshold, anchors, threshold):
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    bboxes = [[] for _ in range(1)]
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    nms_boxes = non_max_suppression(bboxes[0], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
    model.train()
    return nms_boxes

model = YOLOv3(num_classes=config.NUM_CLASSES)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 6))

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    original_img = img
    original_img_np = np.array(img)
    img = config.test_transforms(image=np.array(img))["image"]
    img = img.unsqueeze(0)
    img = img.to(device)
    results = get_bboxes(img, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
    for r in results:
        class_pred, prob_score, center_x, center_y, width, height = r
        if prob_score > 0.95:
            # Convert YOLO format (center_x, center_y, width, height) to pixel coordinates
            img_width, img_height = original_img.size
            x1 = int((center_x - width / 2) * img_width)
            y1 = int((center_y - height / 2) * img_height)
            x2 = int((center_x + width / 2) * img_width)
            y2 = int((center_y + height / 2) * img_height)
            if class_pred == 0:
                cv2.rectangle(original_img_np, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                cv2.rectangle(original_img_np, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Update the figure with the new image
    ax.clear()  # Clear the previous image
    ax.imshow(original_img_np)
    ax.axis("off")
    ax.set_title(f"Prediction: {img_name}")
    plt.draw()  # Redraw the updated image
    plt.pause(5)  # Pause to simulate the video effect, adjust as necessary

    #fig.set_size_inches(original_img_np.shape[1] / 100, original_img_np.shape[0] / 100)  # Scale to 100 DPI
    #ax.clear()  # Clear the previous image
    #ax.imshow(original_img_np)
    #ax.axis("off")
    #ax.set_title(f"Prediction: {img_name}")
    #plt.tight_layout()  # Ensure the image fills the figure
    #plt.draw()  # Redraw the updated image
    #plt.pause(0.0000001)

plt.ioff()  # Turn off interactive mode to stop dynamic updates
