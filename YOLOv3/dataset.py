import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

from utils import (
    iou_width_height as iou,
    non_max_suppression as nms
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        #csv_file_annot,
        csv_file_img,
        img_dir,
        #label_dir,
        anchors,
        image_size=config.IMAGE_SIZE, #416,
        S=[13, 26, 52],
        C=config.NUM_CLASSES,
        transform=None,
    ):
        #self.annotations = pd.read_csv(csv_file_annot)
        self.annotations = pd.read_csv(config.DATASET + config.TRAIN_FILE)
        #self.annotations = pd.read_csv(csv_file_img)
        self.img_dir = img_dir
        #self.label_dir = label_dir
        self.img_names = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        #return len(self.annotations)
        return len(self.img_names)

    def __getitem__(self, index):
        #label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        #bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        bboxes = np.roll(self.annotations[self.annotations['filename']==self.img_names.iloc[index][0]].iloc[:,1:].values, 4, axis=1).tolist()

        #img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img_path = os.path.join(self.img_dir, self.img_names.iloc[index][0])
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                # Assign a bounding box to a specific anchor in the target tensor if the anchor at the given 
                # grid cell is not already taken and the current scale does not already have an assigned anchor.
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
