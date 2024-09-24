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
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Apply Non-Maximum Suppression
    pred_boxes = [non_max_suppression(p, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD) for p in predictions]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for boxes in pred_boxes:
        for box in boxes:
            x1, y1, x2, y2 = box[:4].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save or display the image
    image.show()

# Example usage
img_path = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images\with_mask_3424.jpg'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_8.pth.tar'
draw_boxes(img_path, model_path)

