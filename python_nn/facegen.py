import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromTensorflow("/home/delta8/models/gan/generator.pb", "/home/delta8/models/gan/generator.pbtxt")
cv.namedWindow('Frame', cv.WINDOW_NORMAL)
while cv.waitKey(100) < 0:
    inp = np.random.uniform(-1, 0, [1,64]).astype(np.float32)
    net.setInput(inp)
    out = net.forward()
    out =(out +1)/2
    np.clip(out, 0,1)
    out = np.transpose(out,[0,2,3,1])
    out = np.reshape(out,[64,64,3])
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    cv.imshow('Frame', out)
