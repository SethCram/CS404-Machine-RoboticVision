#!/usr/bin/env python3
"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Assignment 4 - Motion Detector
"""

import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs

if __name__ == "__main__":

    #open webcam
    webcam = cv.VideoCapture(0)

    key = ord('r')

    cv.namedWindow(mv_functs.Impl_Consts.CONTROLS_PANEL_NAME)
        
    cv.createTrackbar("lower", mv_functs.Impl_Consts.CONTROLS_PANEL_NAME, 0, 255, mv_functs.nothing)
    cv.createTrackbar("upper", mv_functs.Impl_Consts.CONTROLS_PANEL_NAME, 0, 255, mv_functs.nothing)

    #show webcam footage
    while (key != mv_functs.Impl_Consts.CLOSE_KEY):
        still = webcam.read()
        
        og_img = still[1]
        
        #should undistortion json info be used? or just for images and not webcame caps?
            
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #gaussian blur
        img = cv.GaussianBlur(img, (5,5), 0)
        
        #get trackbar poses
        lower = int( cv.getTrackbarPos('lower', mv_functs.Impl_Consts.CONTROLS_PANEL_NAME) )
        upper = int( cv.getTrackbarPos('upper', mv_functs.Impl_Consts.CONTROLS_PANEL_NAME) )
        
        #canny edge
        img = cv.Canny(img, lower, upper)
        
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        contours = list(contours)
        contours.sort(key=cv.contourArea, reverse=True)
        contours = contours[0]
        
        cv.drawContours(og_img, contours, -1, (255, 0, 0), 3)
        img = og_img
        
        #bounding rect
        x,y,w,h = cv.boundingRect(contours)
        cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        #rot'd rect (more conservative w/ area and rots to fit shape)
        rect = cv.minAreaRect(contours)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0,0,255), 2)
        
        #min enclosing circ
        (x,y), radius = cv.minEnclosingCircle(contours)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(img, center, radius, (0,255,0), 2)
        
        cv.imshow(mv_functs.Impl_Consts.IMAGE_WINDOW_NAME, img) #first val is data type
        
        key = cv.waitKey(5)
        
    # After the loop release the cap object
    webcam.release()
    cv.destroyAllWindows()
        