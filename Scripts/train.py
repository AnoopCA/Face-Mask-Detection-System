import torch
import torch.nn as nn

import pandas as pd
from PIL import Image
from torchvision import transforms

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
                layers.append(nn.Dropout(p=0.4))
            in_features = out_features
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_data(df, img_dir, img_size=(224, 224)):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    images = []
    targets = []
    for filename in df['filename'].unique():
        img_path = f"{img_dir}\{filename}"
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
        annotations = df[df['filename'] == filename].head(20)
        boxes = []
        labels = []
        
        for _, row in annotations.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            label = 0 if row['label'] == 'without_mask' else 1
            
            boxes.append([xmin / row['width'], ymin / row['height'], xmax / row['width'], ymax / row['height']])
            labels.append(label)
        
        max_detect = 20
        while len(boxes) < max_detect:
            boxes.append([0, 0, 0, 0])
            labels.append(0)  # Assuming -1 means no object
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = torch.cat((boxes, labels.unsqueeze(1)), dim=1)
        targets.append(target)
    
    images = torch.stack(images)
    targets = torch.stack(targets)

    return images, targets

def main():
    df = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')
    img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\images'
    model = FaceMaskDetection().to(device)

    images, targets = load_data(df, img_dir, (256, 256))
    

    dataset = TensorDataset(images, targets)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FaceMaskDetection()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 128
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {test_loss/len(test_loader):.4f}')

    torch.save(model.state_dict(), r"D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_1.pth")

if __name__ == "__main__":
    main()
