import os
import pandas as pd
import cv2


image_path = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\images'
image_name = 'maksssksksss99.png'
image = cv2.imread(os.path.join(image_path, image_name))

annotations = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')
img_table = annotations[annotations['filename']==image_name]

for index, row in img_table.iterrows():
    if (row['label'] == 'with_mask') or (row['label']=='mask_weared_incorrect'):
        x1 = row['xmin']
        y1 = row['ymin']
        x2 = row['xmax']
        y2 = row['ymax'] 
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)

cv2.imshow(image_name, image)
cv2.waitKey(0)

