import os
import torch
import cv2
from torch.utils.data import TensorDataset, DataLoader
from train import FaceMaskDetection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_7.pth'

test_images = TensorDataset(image_dir)
test_data = DataLoader(dataset=test_images, batch_size=32)

model = FaceMaskDetection()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad:
    for inputs in test_data:
        outputs = model(inputs)