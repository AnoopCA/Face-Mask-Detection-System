import cv2
facemodel = cv2.CascadeClassifier("face.xml")
vid = cv2.VideoCapture("group.mp4") #"http://192.168.1.6:8080/video")
i = 1
while(vid.isOpened()):
    flag, frame=vid.read()
    if (flag):
        faces = facemodel.detectMultiScale(frame)
        for (x,y,l,w) in faces:
            face_img = frame[y:y+w, x:x+l]
            path = "data/" + str(i) + ".jpg"
            i = i+1
            cv2.imwrite(path, face_img)
            cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255,0), 3)
        cv2.namedWindow("Face Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Mask", frame)
        key = cv2.waitKey(20)
        if (key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()
