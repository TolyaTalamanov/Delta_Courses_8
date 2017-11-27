import cv2 as cv
import numpy as np

print(cv.__version__)
target_face = cv.imread("test1.jpg")
cap = cv.VideoCapture(0)
net = cv.dnn.readNetFromTorch('/models/openface/nn4.small2.v1.t7')

def return_vector(frame):
    input = cv.dnn.blobFromImage(frame, 1.0/255.0, (96, 96), swapRB=True)
    net.setInput(input)
    return net.forward()

target_vector = return_vector(target_face)[0]
while cv.waitKey(1) < 0:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    coeff = np.dot(target_vector, return_vector(frame)[0])
    print(coeff)

    cv.namedWindow("window")
    cv.imshow("window", frame)

    #cv.imshow("window", frame)
