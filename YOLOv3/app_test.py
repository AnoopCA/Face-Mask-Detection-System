import os
import sys
import torch

sys.path.append(os.path.abspath(r"D:\ML_Projects\Face-Mask-Detection-System\YOLOv3"))
from model import YOLOv3
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = r'D:\ML_Projects\Face-Mask-Detection-System\YOLOv3\Models\fmd_yolov3_10.pth.tar'

model = YOLOv3(num_classes=config.NUM_CLASSES)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
