import cv2 as cv
import numpy as np

models_path = "C:/Users/Drinker/Desktop/models/"

_net_detection = cv.dnn.readNetFromCaffe(models_path + 'face_detection/face_detector_config.prototxt',
                                     models_path + 'face_detection/face_detector_weights.caffemodel')

def draw_faces(frame):
    input = cv.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), crop=False)
    _net_detection.setInput(input)
    out = _net_detection.forward()
    for rect in out[0][0]:
        if rect[2] > 0.7:
            left = int(rect[3]*frame.shape[1])
            top =  int(rect[4]*frame.shape[0])
            right = int(rect[5]*frame.shape[1])
            bottom = int(rect[6]*frame.shape[0])           
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

def get_list_of_faces_and_draw(frame):
    input = cv.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), crop=False)
    _net_detection.setInput(input)
    out = _net_detection.forward()
    face_imgs = []
    for rect in out[0][0]:
        if rect[2] > 0.7:
            left = int(rect[3]*frame.shape[1])
            top =  int(rect[4]*frame.shape[0])
            right = int(rect[5]*frame.shape[1])
            bottom = int(rect[6]*frame.shape[0])
            face_imgs.append(np.copy(frame[max(top , 0):(top + abs(bottom - top)),
                                   max(left , 0):(left + abs(right - left))]))
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    return face_imgs;

