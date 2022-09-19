#!/usr/bin/env python3
import numpy as np
import cv2 as cv
#from MachineVisionLibrary import mv_functs

webcam = cv.VideoCapture(0)

def nothing(x):
    pass


def threshold():
    """
    Adjust webcam threshold dynamically w/ slider.
    """
    key = ord("r")
    cv.namedWindow('controls')
    cv.createTrackbar("threshold", "controls", 0, 255, nothing)
    cv.setTrackbarPos('threshold', 'controls', 127)
    
    while key != ord('s'): #ord = key press
        still = webcam.read()
        #conv to greyscale
        imgGrey = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        thresh = int( cv.getTrackbarPos('threshold', 'controls'))
        ret, imgThreshBin = cv.threshold(imgGrey, thresh, 255, cv.THRESH_BINARY)
        cv.imshow("Image", imgThreshBin)
        key = cv.waitKey(5)

def blur():
    """
    Adjust webcam blur dynamically w/ slider.
    """
    key = ord("r")
    cv.namedWindow('controls')
    cv.createTrackbar("blur kernel", "controls", 3, 100, nothing)
    cv.setTrackbarPos('blur kernel', 'controls', 3)
    
    while key != ord('s'): #ord = key press
        still = webcam.read()
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        blurLevel = int( cv.getTrackbarPos('blur kernel', 'controls'))
        #turn even numbers odd bc odd kernel needed
        if blurLevel % 2 == 0:
            blurLevel += 1
            
        img = cv.GaussianBlur(img, (blurLevel, blurLevel), 0)
        cv.imshow("Image", img)
        key = cv.waitKey(5)     
        
def morph():
    """
    Adjust webcam morphology dynamically w/ mult sliders.
    """
    key = ord("r")
    cv.namedWindow('controls')
    
    #Threshold controls
    cv.createTrackbar("threshold", "controls", 3, 100, nothing)
    cv.setTrackbarPos('threshold', 'controls', 3)
    
    #erosion toggle 
    cv.createTrackbar("erosion toggle", "controls", 0, 1, nothing)
    cv.setTrackbarPos('erosion', 'controls', 0)
    
    #Erosion kernel size controls (if toggle enabled)
    cv.createTrackbar("erosion kernel", "controls", 0, 30, nothing)
    cv.setTrackbarPos('erosion', 'controls', 3)
    
    #Dilation toggle
    cv.createTrackbar("dilation toggle", "controls", 0, 1, nothing)
    cv.setTrackbarPos('dilation', 'controls', 0)
    
    #Dilate kernel size controls (if toggle enabled)
    cv.createTrackbar("dilation kernel", "controls", 0, 30, nothing)
    cv.setTrackbarPos('dilation', 'controls', 3) 
    
    while key != ord('s'): #ord = key press
        still = webcam.read()
        #conv to greyscale
        imgGrey = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        blurLevel = int( cv.getTrackbarPos('blur kernel', 'controls'))
        #turn even numbers odd bc odd kernel needed
        if blurLevel % 2 == 0:
            blurLevel += 1
        #blur (fails)
        imgBlur = cv.GaussianBlur(imgGrey, (7, 7), 0)
        
        #thresh 
        thresh = int(cv.getTrackbarPos('threshold', 'controls'))
        img = cv.threshold(imgBlur, thresh, 255, cv.THRESH_BINARY)
        
        #erosion
        erode = int( cv.getTrackbarPos('erosion toggle', 'controls'))
        if erode:
            erosionLevel = int(cv.getTrackbarPos('erosion kernel', 'controls'))
            if(erosionLevel % 2 == 0):
                erosionLevel += 1
        
            img = cv.erode(img, np.ones((erosionLevel, erosionLevel)), dtype=int)
        
        #dilation
        dilate = int( cv.getTrackbarPos('dilation toggle', 'controls'))
        if dilate:
            dilateLevel = int(cv.getTrackbarPos('dilation kernel', 'controls'))
            if(dilateLevel % 2 == 0):
                dilateLevel += 1
        
            img = cv.dilate(img, np.ones((dilateLevel, dilateLevel)), dtype=int)
        
        #erosion followed by dilaton is opening
        
        cv.imshow("Image", img)
        key = cv.waitKey(5)   

#threshold()
#blur()                
#morph()