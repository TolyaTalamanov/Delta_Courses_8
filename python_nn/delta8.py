import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromTensorflow("D:\delta\models\emotions\emotion_recognition.pb")
cap = cv.VideoCapture(0)
cv.namedWindow('Frame', cv.WINDOW_NORMAL)
emos = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
while cv.waitKey(1) < 0:
    has_frame, frame = cap.read()
    if not has_frame:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blob = cv.dnn.blobFromImage(gray_frame, 1.0/255, (48, 48))

    net.setInput(blob)
    out = net.forward()
    i = np.argmax(out[0])
    print emos[i]
    cv.imshow('Frame', frame)
