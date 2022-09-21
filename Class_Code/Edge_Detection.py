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

def basic_canny():
    key = ord('r')
    img = cv.imread('MeAndEmoly.jpg')
    
    #conv to greyscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #gaussian blur
    img = cv.GaussianBlur(img, (5,5), 0)
    
    img = cv.Canny(img, 5, 200)
    
    #show image
    while key != ord('s'):
        cv.imshow("image", img)
        key = cv.waitKey()

def trackbar_canny():
    key = ord('r')
    cv.namedWindow('controls')
    
    cv.createTrackbar("lower", 'controls', 0, 255, nothing)
    cv.createTrackbar("upper", 'controls', 0, 255, nothing)
    
    while key != ord('s'):
        still = webcam.read()
        
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #gaussian blur
        img = cv.GaussianBlur(img, (5,5), 0)
        
        #get trackbar poses
        lower = int( cv.getTrackbarPos('lower', 'controls'))
        upper = int( cv.getTrackbarPos('upper', 'controls'))
        
        #canny edge
        img = cv.Canny(img, lower, upper)
        
        cv.imshow("image", img)
        key = cv.waitKey(5)

def auto_canny():
    key = ord('r')
    cv.namedWindow('controls')
    
    cv.createTrackbar("lower", 'controls', 0, 255, nothing)
    cv.createTrackbar("upper", 'controls', 0, 255, nothing)
    
    while key != ord('s'):
        still = webcam.read()
        
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        v = np.median(img)
        
        #gaussian blur
        img = cv.GaussianBlur(img, (5,5), 0)
        
        #get trackbar poses
        #lower = int( cv.getTrackbarPos('lower', 'controls'))
        #upper = int( cv.getTrackbarPos('upper', 'controls'))
        
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma)*v))
        upper = int(max(255, (1.0 + sigma)*v))
        
        #canny edge
        img = cv.Canny(img, lower, upper)
        
        cv.imshow("image", img)
        key = cv.waitKey(5)

def contours():
    key = ord('r')
    cv.namedWindow('controls')
    
    cv.createTrackbar("lower", 'controls', 0, 255, nothing)
    cv.createTrackbar("upper", 'controls', 0, 255, nothing)
    
    while key != ord('s'):
        still = webcam.read()
        
        og_img = still[1]
        
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #v = np.median(img)
        
        #gaussian blur
        img = cv.GaussianBlur(img, (5,5), 0)
        
        #get trackbar poses
        lower = int( cv.getTrackbarPos('lower', 'controls'))
        upper = int( cv.getTrackbarPos('upper', 'controls'))
        
        #sigma = 0.33
        #lower = int(max(0, (1.0 - sigma)*v))
        #upper = int(max(255, (1.0 + sigma)*v))
        
        #canny edge
        img = cv.Canny(img, lower, upper)
        
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(og_img, contours, -1, (255, 0, 0), 3)
        img = og_img
        
        cv.imshow("image", img)
        key = cv.waitKey(5)

def shape_contours():
    """
    Not good at background and foreground determination.
    """
    key = ord('r')
    cv.namedWindow('controls')
    
    cv.createTrackbar("lower", 'controls', 0, 255, nothing)
    cv.createTrackbar("upper", 'controls', 0, 255, nothing)
    
    while key != ord('s'):
        still = webcam.read()
        
        og_img = still[1]
        
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        #v = np.median(img)
        
        #gaussian blur
        img = cv.GaussianBlur(img, (5,5), 0)
        
        #get trackbar poses
        lower = int( cv.getTrackbarPos('lower', 'controls'))
        upper = int( cv.getTrackbarPos('upper', 'controls'))
        
        #sigma = 0.33
        #lower = int(max(0, (1.0 - sigma)*v))
        #upper = int(max(255, (1.0 + sigma)*v))
        
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
        
        cv.imshow("image", img)
        key = cv.waitKey(5)



#sobel()

#basic_canny()

#trackbar_canny()

#auto_canny()

#contours()

shape_contours()