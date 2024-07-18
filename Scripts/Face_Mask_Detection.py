import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Face_Mask_Detection(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.classifer = nn.Sequential(

        )

    def forward(self):
        pass

Face_Mask_Detection = Face_Mask_Detection().to(device)
