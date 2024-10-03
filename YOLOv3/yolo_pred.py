import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from model import YOLOv3
from utils import non_max_suppression, cells_to_bboxes
import config
import cv2

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

#   There are (1 * 3 * 7 * 7) * (1 * 3 * 14 * 14) * (1 * 3 * 28 * 28) bounding boxes of 7 elements in each before the above preprocessing
#   There are 3087 bounding boxes of 7 elements, in a list after the above preprocessing
#   While checking the bounding boxes sent by the train.py, it is 6 dimensional not 7 which is conflicting with the above
#   Make use of the logic in "get_evaluation_bboxes" function before applying non_max_suppression

# *************************************************************************************************************

    iou_threshold=config.NMS_IOU_THRESH
    anchors=config.ANCHORS
    threshold=config.CONF_THRESHOLD
    train_idx = 0
    all_pred_boxes = []

    batch_size = image.shape[0]
    bboxes = [[] for _ in range(batch_size)]
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(
            predictions[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format="midpoint",
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)
        train_idx += 1

    print(f"all_pred_boxes: {all_pred_boxes}")

# *************************************************************************************************************

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(Image.open(image_path).convert("RGB"))
    #for boxes in all_pred_boxes:
    for box in all_pred_boxes:
        x1, y1, x2, y2 = box[:4] #.tolist()
        print(f"x1: {x1}")
        print(f"y1: {y1}")
        print(f"x2: {x2}")
        print(f"y2: {y2}")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Display the image
    open_cv_image = np.array(Image.open(image_path).convert("RGB"))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Display the modified image using OpenCV
    cv2.imshow('Image with Rectangles', open_cv_image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

img_path = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images\with_mask_3424.jpg'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_9.pth.tar'
draw_boxes(img_path, model_path)
