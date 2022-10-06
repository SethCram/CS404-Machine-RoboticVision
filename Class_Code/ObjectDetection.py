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
        #prev, nxt, flow, pyr_scale, levels, etc.
        flow = cv.calcOpticalFlowFarneback(prevs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        #Hue to angle
        hsv[..., 0] = ang * 180 / np.pi / 2
        #Value to magnitude 
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow("image", img)
        key = cv.waitKey(5)
        prev = nxt
        
def harris_corners():
    """
    Marks on a pixel by pixel basis
    """
    key = ord('r')
    webcam = cv.VideoCapture(0)
    
    while key != ord('s'):
        still = webcam.read()
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #conv to float
        img = np.float32(img)
        img = cv.cornerHarris(img, 2, 3, 0.04)
        img = cv.dilate(img, None)
        
        #threshold
        still[1][img > 0.01*img.max()] = [0, 0, 255] #mask to show red
        img = still[1]
        cv.imshow("image", img)
        key = cv.waitKey(5)
        
def shi():
    key = ord('r')
    webcam = cv.VideoCapture(0)
    
    while key != ord('s'):
        still = webcam.read()
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #conv to float
        img = np.float32(img)
        corners = cv.goodFeaturesToTrack(img, maxCorners= 100, qualityLevel= 0.01, minDistance=10 )
        corners = np.int0(corners)
        img = still[1]
        #formatting to draw circles on each of corners
        for i in corners:
            x,y = i.ravel()
            cv.circle(img, (x,y), 5, 255, -1)
        
        cv.imshow("image", img)
        key = cv.waitKey(5)

def SIFT():
    webcam = cv.VideoCapture(0)
    key = ord("r")
    while key != ord(mv_functs.Impl_Consts.CLOSE_KEY):
        still = webcam.read()
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #create sift
        sift = cv.SIFT_create()
        
        #detect keypoints
        kp = sift.detect(img, None) #2nd arg = optional mask 
        
        #draw keypoints
        img = cv.drawKeypoints(img, kp, outImage=None)
        
        cv.imshow("image", img)
        
        key = cv.waitKey(5)
   
def BRISK():
    webcam = cv.VideoCapture(0)
    key = ord('r')
    
    while key != ord(mv_functs.Impl_Consts.CLOSE_KEY):
        still = webcam.read()
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #create brisk obj
        brisk = cv.BRISK_create()
        #detect keypoints
        kp = brisk.detect(img, None)
        #draw keypoints
        img = cv.drawKeypoints(img, kp, outImage=None)
        
        cv.imshow("image", img)
        key = cv.waitKey(5)
 
def blob():
    webcam = cv.VideoCapture(0)
    key = ord('r')
    
    while key != ord(mv_functs.Impl_Consts.CLOSE_KEY):
        still = webcam.read()
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #create new obj
        blob = cv.SimpleBlobDetector_create()
        #detect keypoints
        kp = blob.detect(img, None)
        #draw keypoints
        #img = cv.drawKeypoints(img, kp, outImage=None)
        img = cv.drawKeypoints(img, kp, outImage=None, color= (0, 0, 255), flags = cv.DRAW_MATCHES_FLAGS_DEFAULT)
        
        cv.imshow("image", img)
        key = cv.waitKey(5)

def BruteForce_Matcher():
    
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
    
#dense_OpticalFlow()

#harris_corners()

#shi()
    
#SIFT()

#BRISK()

#blob()

BruteForce_Matcher()