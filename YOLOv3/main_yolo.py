# Import necessary libraries
import numpy as np
import os
import sys
import torch
import cv2
from PIL import Image
import streamlit as st
import tempfile

# Add the specified directory to the system path to enable importing modules from the YOLOv3 project folder
sys.path.append(os.path.abspath(r"D:\ML_Projects\Face-Mask-Detection-System\YOLOv3"))
from model import YOLOv3
import config
from utils import cells_to_bboxes, non_max_suppression

# Setup GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_12.pth.tar'

# Function to get bounding boxes from model predictions with non-max suppression applied
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

# Load the YOLOv3 model with pre-trained weights and set it to evaluation mode
model = YOLOv3(num_classes=config.NUM_CLASSES)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# Object Detection and Bounding Box Visualization Using YOLO Predictions
def get_pred(raw_img, bb_ln_width):
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

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
                cv2.rectangle(original_img_np, (x1, y1), (x2, y2), (0, 255, 0), bb_ln_width)
            else:
                cv2.rectangle(original_img_np, (x1, y1), (x2, y2), (255, 0, 0), bb_ln_width)
    return(original_img_np)


st.title("Face Mask Detection System")
choice = st.sidebar.selectbox("MENU", ("HOME", "IMAGE", "VIDEO", "CAMERA"))

if choice == "HOME":
    st.header("Welcome!")

elif choice == "IMAGE":
    file = st.file_uploader("Upload Image")
    if file:
        b = file.getvalue()
        d = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(d, cv2.IMREAD_COLOR)
        img = get_pred(img, 2)
        st.image(img, channels='RGB', width=400)

elif choice == "VIDEO":
    file = st.file_uploader("Upload Video")
    windows = st.empty()
    if file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        vid = cv2.VideoCapture(temp_file.name)
        while vid.isOpened():
            flag, frame = vid.read()
            if not flag:
                break
            img = get_pred(frame, 2)
            windows.image(img, channels="RGB")
        vid.release()

elif choice == "CAMERA":
    st.session_state["CAMERA"] = True
    k = st.text_input("Enter 0 to open webcam or write URL for opening IP camera")
    if len(k) == 1:
        k = int(k)
    btn = st.button("Start Camera")
    cls_btn = st.button("Stop Camera")
    if cls_btn:
        st.session_state["CAMERA"] = False
    windows = st.empty()
    if btn and st.session_state["CAMERA"]:
        vid = cv2.VideoCapture(k)
        while(vid.isOpened()):
            flag, frame=vid.read()
            if (flag):
                img = get_pred(frame, 2)
                windows.image(img, channels="RGB")
        vid.release()