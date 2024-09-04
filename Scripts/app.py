import numpy as np
import os
import torch
import cv2
from torchvision import transforms
from PIL import Image
from train import FaceMaskDetection
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.04

#img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_images'
img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\to test_images - Other Images'
#img_out = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\test_output'
model_path = r'D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_12.pth'
#model_path = r'D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_13_e128_No_AvgPool.pth'
#model_path = r'D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_14_e128_AvgPool.pth'

model = FaceMaskDetection()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
                                 transforms.Resize((224, 224)),
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
    results = model(img).to(device)
    for result in results:
        for r in result:
            x1, y1, x2, y2, score = r
            if score > 0.001:
                x1 = int(x1 * original_img.shape[1])
                y1 = int(y1 * original_img.shape[0])
                x2 = int(x2 * original_img.shape[1])
                y2 = int(y2 * original_img.shape[0])
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("Prediction", original_img)
    cv2.waitKey(2000)

    #output_img_path = os.path.join(img_out, f"output_{img_name}")
    #cv2.imwrite(output_img_path, original_img)
