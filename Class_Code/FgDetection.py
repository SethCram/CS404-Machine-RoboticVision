#!/usr/bin/env python3
"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Class Notes
"""

import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs

#key = ord('r')

#img = cv.imread('MeAndEmoly.jpg')

def connected():
    img = cv.imread('MeAndEmoly.jpg')
    og_img = img.copy()
    
    #greyscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #threshold w/ Otsu's using a histogram
    ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    
    #apply connected comps funct
    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(img, 8, cv.CV_32S)
    
    for i in range(0, numLabels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        cv.rectangle(og_img, (x,y), (x+w, y+h), (0, 255, 0), 3)
        
    #key = ord('r')
    mv_functs.showImage(og_img)
        
def GrabCut():
    key = ord('r')
    img = cv.imread('MeAndEmoly.jpg')
    
    #resize image
    img = cv.resize(img, (int(img.shape[1]*0.20), int(img.shape[0]*0.20)), interpolation=cv.INTER_AREA)
    
    #make mask same shape as img
    mask = np.zeros(img.shape[:2], np.uint8)
    
    #common vals for fg and bg temp models
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    
    #better rect for segmenting
    rect = (133, 185, 394, 489)
    cv.grabCut(img, mask, rect, bgModel, fgModel, 3, cv.GC_INIT_WITH_RECT)
    
    #mask
    mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    mv_functs.showImage(img)
        
def BgSubtractor():
    webcam = cv.VideoCapture(0)
    
    key = ord('r')
    
    #bg = cv.createBackgroundSubtractorKNN()
    bg = cv.createBackgroundSubtractorMOG2()
    
    while key != ord('s'):
        still = webcam.read()
        
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #could blur too
    
    
        img = bg.apply(img)
        cv.imshow("Image", img)
        key = cv.waitKey(5)
        
        
        
#connected()
#GrabCut()

BgSubtractor()