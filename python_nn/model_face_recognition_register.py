# It script used for register faces, after add beter if you change names on yours in face_config.json
import cv2 as cv
import numpy as np
import time
import json
import model_face_detection

cap = cv.VideoCapture(0)

models_path = "C:/Users/Drinker/Desktop/models/"

net_recogn = cv.dnn.readNetFromTorch(models_path + 'openface/nn4.small2.v1.t7')

def return_vector(frame):
    input = cv.dnn.blobFromImage(frame, 1.0/255.0, (96, 96), swapRB=True)
    net_recogn.setInput(input)
    return net_recogn.forward()[0]

try:
    with open('face_config.json', 'r') as f:
        faces = json.load(f)
except Exception:
	faces = []

while cv.waitKey(1) < 0:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    face_imgs = model_face_detection.get_list_of_faces_and_draw(frame)   
    cv.imshow("window", frame)

ind = 0

for img in face_imgs:
    cv.imshow("unregistered" + str(ind), img)
    vec = return_vector(img).tolist()
    faces.append({"vector": vec, "name": ("unregistered" + str(ind))})
    ind = ind + 1

with open('face_config.json', 'w') as f:
    json.dump(faces, f)

while cv.waitKey(1) < 0:
    dosth = 1


