import cv2 as cv
import numpy as np
import json

# print(cv.__version__)

models_path = "C:/Users/Drinker/Desktop/models/"

try:
    with open('face_config.json', 'r') as f:
        faces = json.load(f)
except Exception:
    raise RuntimeError('There are not registered faces')

# cap = cv.VideoCapture(0)
net = cv.dnn.readNetFromTorch(models_path + 'openface/nn4.small2.v1.t7')

def return_vector(frame):
    input = cv.dnn.blobFromImage(frame, 1.0/255.0, (96, 96), swapRB=True)
    net.setInput(input)
    return net.forward()[0]


def get_most_possble_face(frame):
    coefs = []
    curr_vec = return_vector(frame)
    for face in faces:
        coefs.append({"name" : face["name"], "coeff" : np.dot(face["vector"], curr_vec)})    
    coefs = sorted(coefs, key=lambda k: k["coeff"])
    return coefs[len(coefs) - 1]

"""while cv.waitKey(1) < 0:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    print(get_most_possble_face(frame))

    cv.namedWindow("window")
    cv.imshow("window", frame)"""
