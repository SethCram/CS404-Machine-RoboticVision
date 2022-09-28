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

img = cv.imread("MeAndEmoly.jpg")

def colormap():
    webcam = cv.VideoCapture(0)
    
    key = ord('r')
    
    while key != ord("s"):
    
        still = webcam.read()
        
        #conv greyscale 
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        img = cv.applyColorMap(img, cv.COLORMAP_TWILIGHT)
        
        cv.imshow("press s to close", img)
        key = cv.waitKey(5)
     
def pyramids():
    
    img = cv.imread("MeAndEmoly.jpg")
    layer = img.copy()
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(layer)
        cv.imshow(str(i), layer)
        cv.waitKey(500)

def tut_pyramids():
    # Load the image
    src = cv.imread(cv.samples.findFile("MeAndEmoly.jpg"))
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: pyramids.py [image_name -- default ../data/chicky_512.png] \n')
        return -1
    
    while 1:
        rows, cols, _channels = map(int, src.shape)
        
        cv.imshow('Pyramids Demo', src)
        
        k = cv.waitKey(0)
        if k == 27:
            break
            
        elif chr(k) == 'i':
            src = cv.pyrUp(src, dstsize=(2 * cols, 2 * rows))
            print ('** Zoom In: Image x 2')
            
        elif chr(k) == 'o':
            src = cv.pyrDown(src, dstsize=(cols // 2, rows // 2))
            print ('** Zoom Out: Image / 2')
            
        
#colormap()

#pyramids()

tut_pyramids()