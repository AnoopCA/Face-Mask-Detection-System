import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import tempfile

facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("D:/ML_Projects/Face-Mask-Detection-System/Models/mask_model.h5")

st.title("Face Mask Detection System")
choice = st.sidebar.selectbox("MENU", ("HOME", "IMAGE", "VIDEO", "CAMERA"))

if choice == "HOME":
    st.header("Welcome!")

elif choice == "IMAGE":
    file = st.file_uploader("Upload Image")
    if file:
        b = file.getvalue()
        d = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(d, cv2.IMREAD_COLOR)
        faces = facemodel.detectMultiScale(img)
        for (x,y,l,w) in faces:
            crop_face_1 = img[y:y+w, x:x+l]
            cv2.imwrite('D:/ML_Projects/Face-Mask-Detection-System/Archives/temp.jpg', crop_face_1)
            crop_face = load_img('D:/ML_Projects/Face-Mask-Detection-System/Archives/temp.jpg', target_size=(150,150,3))
            crop_face = img_to_array(crop_face)
            crop_face = np.expand_dims(crop_face, axis=0)
            pred = maskmodel.predict(crop_face)[0][0]
            if pred == 1:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,0,255), 4)
            else:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,255, 0), 4)
        st.image(img, channels='BGR', width=400)

elif choice == "VIDEO":
    file = st.file_uploader("Upload Video")
    windows = st.empty()
    if file:
        vid = cv2.VideoCapture(file.name)
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
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255, 0), 4)
                    windows.image(frame, channels="BGR")

elif choice == "CAMERA":
    st.session_state["CAMERA"] = True
    k = st.text_input("Enter 0 to open webcam or write URL for opening IP camera")
    if len(k) == 1:
        k = int(k)
    btn = st.button("Start Camera")
    cls_btn = st.button("Stop Camera")
    if cls_btn:
        st.session_state["CAMERA"] = False
    windows = st.empty()
    if btn and st.session_state["CAMERA"]:
        vid = cv2.VideoCapture(k)
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
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255, 0), 4)
                    windows.image(frame, channels="BGR")
        