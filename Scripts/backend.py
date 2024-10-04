import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("D:/ML_Projects/Face-Mask-Detection-System/Models/mask_model.h5")
vid = cv2.VideoCapture("group.mp4") #"http://192.168.1.6:8080/video")
i = 1
while(vid.isOpened()):
    flag, frame=vid.read()
    if (flag):
        faces = facemodel.detectMultiScale(frame)
        for (x,y,l,w) in faces:
            crop_face_1 = frame[y:y+w, x:x+l]
            cv2.imwrite('D:/ML_Projects/Face-Mask-Detection-System/Archives/temp.jpg', crop_face_1)
            crop_face = load_img('D:/ML_Projects/Face-Mask-Detection-System/Archives/temp.jpg', target_size=(150,150,3))
            crop_face = img_to_array(crop_face)
            crop_face = np.expand_dims(crop_face, axis=0)
            pred = maskmodel.predict(crop_face)[0][0]
            if pred == 1:
                cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255), 4)
                path = "D:/ML_Projects/Face-Mask-Detection-System/Data/pred_data/" + str(i) + ".jpg"
                cv2.imwrite(path, crop_face_1)
                i = i+1
            else:
                cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255, 0), 4)
        cv2.namedWindow("Face Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Mask", frame)
        key = cv2.waitKey(1)
        if (key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()
