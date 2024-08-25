import torch
import torch.nn as nn

import pandas as pd
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceMaskDetection(nn.Module):
    def __init__(self):
        super(FaceMaskDetection, self).__init__()
        in_channels = 3
        feat_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        in_features = 512 * 7 * 7
        max_detect = 20
        classifier_config = [4096, 4096, max_detect]
        
        self.features = self._make_features(in_channels, feat_config)
        self.classifier = self._make_classifier(in_features, classifier_config, max_detect)

    def _make_features(self, in_channels, config):
        layers = []
        for layer in config:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = layer
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)
    
    def _make_classifier(self, in_features, config, max_detect):
        layers = []
        for out_features in config:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU(inplace=True))
            if out_features != max_detect:
                layers.append(nn.Dropout(p=40))
            in_features = out_features
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = FaceMaskDetection().to(device)
print(model)


def load_data(df, img_dir, img_size=(224, 224)):

    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    images = []
    targets = []
    for filename in df['filename'].unique():
        img_path = f"{img_dir}/{filename}"
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
        annotations = df[df['filename'] == filename]
        boxes = []
        labels = []
        
        for _, row in annotations.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            label = 0 if row['name'] == 'without_mask' else 1
            
            boxes.append([xmin / row['width'], ymin / row['height'], xmax / row['width'], ymax / row['height']])
            labels.append(label)
        
        max_detect = 20
        while len(boxes) < max_detect:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)  # Assuming -1 means no object
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = torch.cat((boxes, labels.unsqueeze(1)), dim=1)
        
        targets.append(target)
    
    images = torch.stack(images)
    targets = torch.stack(targets)

    return images, targets

df = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')
img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\images'
images, targets = load_data(df, img_dir)
