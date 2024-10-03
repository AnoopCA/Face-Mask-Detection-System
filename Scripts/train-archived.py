import datetime
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_DETECT = 20
NUM_EPOCHS = 1024

start_time = datetime.datetime.now()

class FaceMaskDetection(nn.Module):
    def __init__(self):
        super(FaceMaskDetection, self).__init__()
        in_channels = 3
        feat_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        #feat_config = [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        in_features = 512 * 7 * 7
        classifier_config = [4096, 4096, MAX_DETECT * 5]
        #classifier_config = [4096, 4096, 2048, 1024, MAX_DETECT * 5]
        
        self.features = self._make_features(in_channels, feat_config)
        self.classifier = self._make_classifier(in_features, classifier_config, MAX_DETECT)

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
            if out_features != max_detect*5:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=0.5))
            in_features = out_features
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), MAX_DETECT, 5)
        return x

def load_data(df, img_dir, img_size=(224, 224)):
    transform = transforms.Compose([
                                      transforms.Resize(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  ])

#                                      transforms.RandomHorizontalFlip(p=0.5),
#                                      transforms.RandomVerticalFlip(p=0.5),
#                                      transforms.RandomRotation(degrees=30),
#                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#                                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
#                                      transforms.RandomGrayscale(p=0.1),
#                                      transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

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
            if row['label'] != 'without_mask':
                boxes.append([xmin / row['width'], ymin / row['height'], xmax / row['width'], ymax / row['height']])
            else:
                boxes.append([0, 0, 0, 0])
            labels.append(label)

        while len(boxes) < MAX_DETECT:
            boxes.append([0, 0, 0, 0])
            labels.append(0)
        
        boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
        target = torch.cat((boxes, labels.unsqueeze(1)), dim=1)
        targets.append(target)
    
    images = torch.stack(images).to(device)
    targets = torch.stack(targets).to(device)

    return images, targets

def main():
    df = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')
    img_dir = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\images'
    
    images, targets = load_data(df, img_dir, (224, 224))
    dataset = TensorDataset(images, targets)
    train_size = int(0.97 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FaceMaskDetection().to(device)
    #criterion = nn.MSELoss().to(device)
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 0.001

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        #if (epoch+1) % 16 == 0:
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {test_loss/len(test_loader):.4f}')

    torch.save(model.state_dict(), r"D:\ML_Projects\Face-Mask-Detection-System\Models\fmd_17_e2048.pth")

if __name__ == "__main__":
    main()
    print(f"Processing time: {datetime.datetime.now()-start_time}")
