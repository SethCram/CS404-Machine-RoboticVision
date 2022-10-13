#!/usr/bin/env python3
"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Assignment 4 - Motion Detector
"""

import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs
import matplotlib.pyplot as plt

key = ord('r')
    
#read in greyscaled x images of same obj in diff pos's/orientations
img1 = cv.imread("dogSitting.PNG", 0)
img2 = cv.imread("dogWalking.PNG", 0) #should also be img1?

#resize image by half
img1 = cv.resize( img1, (int(img1.shape[1]*0.5), int(img1.shape[0]*0.5)) )
img2 = cv.resize( img2, (int(img2.shape[1]*0.5), int(img2.shape[0]*0.5)) )

orb = cv.ORB_create()
kp1, descr1 = orb.detectAndCompute(img1, None)
kp2, descr2 = orb.detectAndCompute(img2, None) #should also be descr1?

bruteForce = cv.BFMatcher_create(cv.NORM_HAMMING)
matches = bruteForce.match(descr1, descr2)
matches = sorted(matches, key=lambda x:x.distance)

#Draw first 20 matches (could also use KNN)
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[0:20], flags=2, outImg=None)

while key != ord(mv_functs.Impl_Consts.CLOSE_KEY):
    
    cv.imshow("image", img3)
    
    key = cv.waitKey(5)
