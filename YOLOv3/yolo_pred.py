import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from model import YOLOv3
from utils import non_max_suppression, cells_to_bboxes
import config

def draw_boxes(image_path, model_path, device=config.DEVICE):
    # Load the model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load and preprocess the image
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = config.test_transforms
    augmentations = transform(image=image)
    image = augmentations["image"]
    image = image.unsqueeze(0)
    image = image.to(config.DEVICE)

    # Get predictions
    with torch.no_grad():
        predictions = model(image)
    # Apply Non-Maximum Suppression
    #print(f"predictions: {predictions}")
    print(f"predictions: {len(predictions)}")
    print(f"predictions[0]: {len(predictions[0])}")
    print(f"predictions[1]: {len(predictions[0][0])}")
    print(f"predictions[2]: {len(predictions[0][0][0])}")
    print(f"predictions[3]: {len(predictions[0][0][0][0])}")
    print(f"predictions[4]: {len(predictions[0][0][0][0][0])}")
    print(f"predictions[4] data: {predictions[0][0][0][0][0]}")

    pred_boxes = [non_max_suppression(p, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD) for p in predictions]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for boxes in pred_boxes:
        for box in boxes:
            x1, y1, x2, y2 = box[:4].tolist()
            print(f"x1: {x1}")
            print(f"y1: {y1}")
            print(f"x2: {x2}")
            print(f"y2: {y2}")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Display the image
    image.show()


img_path = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images\with_mask_3424.jpg'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_9.pth.tar'
draw_boxes(img_path, model_path)
