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
    
    prevFrame = None

    cv.namedWindow(mv_functs.Impl_Consts.CONTROLS_PANEL_NAME)
        
    threshValStr = "Threshold"
    mv_functs.CreateAndSetTrackbar(threshValStr, initValue=20)
    
    blurKernelStr = "blur kernel"
    mv_functs.CreateAndSetTrackbar(blurKernelStr, initValue=50, lowerBound=3, upperBound=100)
    
    dilationKernelStr = "dilation kernel"
    mv_functs.CreateAndSetTrackbar(dilationKernelStr, initValue=50, lowerBound=3, upperBound=100)
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(webcam.get(3))
    frame_height = int(webcam.get(4))
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    out = cv.VideoWriter('MotionDetectorTest.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    
    #show webcam footage
    while (True):
        still = webcam.read()
        
        og_img = still[1]
        
        #should undistortion json info be used? or just for images and not webcame caps?
            
        #conv to greyscale
        greyFrame = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        blurLevel = int( cv.getTrackbarPos(blurKernelStr, mv_functs.Impl_Consts.CONTROLS_PANEL_NAME))
        #turn even numbers odd bc odd kernel needed
        if blurLevel % 2 == 0:
            blurLevel += 1
        
        #gaussian blur
        blurredFrame = cv.GaussianBlur(greyFrame, (blurLevel, blurLevel), 0)
        
        #if 1st image bc prev frame empty
        if( prevFrame is None):
            #set prev frame to this one
            prevFrame = blurredFrame
            
            #break out of loop + exec nxt iteration
            continue 
        
        #calc diff tween prev and curr frame
        frameDiff = cv.absdiff(prevFrame, blurredFrame)
        
        #update prev frame
        prevFrame = blurredFrame
        
        #dilate img so diffs more visible (better for contour detection)
        dilationKernelInt = int( cv.getTrackbarPos(dilationKernelStr, mv_functs.Impl_Consts.CONTROLS_PANEL_NAME) )
        if dilationKernelInt % 2 == 0:
            dilationKernelInt += 1
        
        dilationKernel = np.ones((dilationKernelInt, dilationKernelInt))
        dilatedFrameDiff = cv.dilate(frameDiff, dilationKernel, 1)
        
        thresh = int( cv.getTrackbarPos(threshValStr, mv_functs.Impl_Consts.CONTROLS_PANEL_NAME) )
        
        #take diff tween areas diff enough (want big area for easier contouring)
        threshFrame = cv.threshold(dilatedFrameDiff, thresh=thresh, maxval=255, type=cv.THRESH_BINARY)[1]
        
        contours, _ = cv.findContours(threshFrame, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        
        #cv.drawContours(image=og_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        
        #if contours found
        if(contours is not None and len(contours) > 0):
            #conv to list to sort by contour area
            contours = list(contours)
            contours.sort(key=cv.contourArea, reverse=True)
            
            #only take biggest contour
            contours = contours[0] 
        
            #bounding rect
            x,y,w,h = cv.boundingRect(contours)
            cv.rectangle(og_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        """
        #walk thru all contours
        for contour in contours:
            #only use contour if big enough
            if( cv.contourArea(contour) >= 200 ):
                #bounding rect
                x,y,w,h = cv.boundingRect(contour)
                cv.rectangle(og_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        """
        # Write the frame into the file 
        out.write(og_img)
        
        cv.imshow(mv_functs.Impl_Consts.IMAGE_WINDOW_NAME, og_img) #first val is data type
        
        if cv.waitKey(5) & 0xFF == ord(mv_functs.Impl_Consts.CLOSE_KEY):
            break
    
    # Release the vid capture object
    webcam.release()
    
    # Release video write object
    out.release()
    
    #close all frames
    cv.destroyAllWindows()
        