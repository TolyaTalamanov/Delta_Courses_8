import cv2 as cv
cap = cv.VideoCapture(0)
net = cv.dnn.readNetFromCaffe('/models/face_detection/face_detector_config.prototxt',
                              '/models/face_detection/face_detector_weights.caffemodel')

while cv.waitKey(1) < 0:
    has_frame, frame = cap.read()
    if not has_frame:
        break
    #print(frame.shape[0])
    #print(frame.shape[1])

    input = cv.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), crop=False)
    net.setInput(input)
    out = net.forward()

    for rect in out[0][0]:
        if rect[2] > 0.7:
            left = int(rect[3]*frame.shape[1])
            top =  int(rect[4]*frame.shape[0])
            right = int(rect[5]*frame.shape[1])
            bottom = int(rect[6]*frame.shape[0])
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv.imshow("window", frame)

    #cv.imshow("window", frame)
