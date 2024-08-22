import os
import pandas as pd
import cv2


image_path = r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\images'
image_name = 'maksssksksss737.png'
image = cv2.imread(os.path.join(image_path, image_name))

target_size = (224, 224)
original_height, original_width = image.shape[:2]
x_scale = target_size[0] / original_width
y_scale = target_size[1] / original_height

resized_image = cv2.resize(image, target_size)
annotations = pd.read_csv(r'D:\ML_Projects\Face-Mask-Detection-System\Data\Kaggle_2\annotations.csv')
img_table = annotations[annotations['filename'] == image_name]

for index, row in img_table.iterrows():
    if row['label'] in ['with_mask', 'mask_weared_incorrect']:
        x1 = int(row['xmin'] * x_scale)
        y1 = int(row['ymin'] * y_scale)
        x2 = int(row['xmax'] * x_scale)
        y2 = int(row['ymax'] * y_scale)
        
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), (255, 0, 255), 1)

cv2.imshow(image_name, resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *** Update this resizing (both image and bounding boxes) to the train.py file *** #

