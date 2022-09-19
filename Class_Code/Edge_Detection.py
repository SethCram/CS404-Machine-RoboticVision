#!/usr/bin/env python3
import numpy as np
import cv2 as cv
#from MachineVisionLibrary import mv_functs

webcam = cv.VideoCapture(0)

def nothing(x):
    pass

def sobel():
    img = cv.imread('MeAndEmoly.jpg')
    key = ord('r')
    
    #conv to greyscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #gaussian blur
    img = cv.GaussianBlur(img, (5,5), 0)
    
    #Sobel deriv 
    # ksize = kernel size
    img = cv.Sobel(img, -1, 1, 1, ksize=5)
    
    #show image
    while key != ord('s'):
        cv.imshow("image", img)
        key = cv.waitKey()

sobel()