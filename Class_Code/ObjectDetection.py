#!/usr/bin/env python3
"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Class Notes
"""

import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs
import matplotlib.pyplot as plt

#img = cv.imread("MeAndEmoly.jpg")

def dense_OpticalFlow():
    webcam = cv.VideoCapture(0)
    
    key = ord('r')
    still = webcam.read()
    prevs = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(still[1])
    hsv[...,1] = 255
    
    while key != ord('s'):
        #could conv to black and white then do erosion for better detection
        
        still = webcam.read()
        nxt = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow("image", img)
        key = cv.waitKey(5)
        prev = nxt
        
dense_OpticalFlow()
