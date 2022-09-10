"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Assignment 3 - Camera Calibration
"""

import os
import numpy as np
import cv2 as cv
import glob
import json
from MachineVisionLibrary import mv_functs

decision = input("Do you need to take more pictures for calibration?(y/n)")

#glob found images
#images = glob.glob(saveDir + '*.jpg')
images = glob.glob('*.jpg')

#if needa take more pics
if(decision == "y"):

    #GET WEB CAM CAPTURES from https://stackoverflow.com/a/34588758/13046931

    #grab device 0 (TAKES A LONG TIME) (works on Windows+Mac)
    webcam = cv.VideoCapture(0)

    #want pattern board to be in various angles for this
    # displays the camera feed in a cv2.namedWindow and will take a snapshot when you hit SPACE. It will also quit if you hit ESC.

    cv.namedWindow("test")
    
    #don't accidently overwrite old images by giving new ones the same names
    if( images != None ):
        img_counter = len(images)
    else:
        img_counter = 0

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("Press esc to close, or space to take a picture", frame)

        k = cv.waitKey(1)
        
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            
            #SAVE IMAGE
            img_name = "opencv_frame_{}.jpg".format(img_counter)
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
        
    webcam.release()
    cv.destroyAllWindows()
    
    #glob prev found images w/ new ones too
    images = glob.glob('*.jpg')

# rest mainly from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html unless otherwise specified

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

patternsCaptured = 0

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#for fname in images:
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        
        #if added more images
        if(decision == "y"):
            #show all images drawn on
            mv_functs.showImage(img)
        
        patternsCaptured += 1
    #if chess board corners not found
    else:
        #remove saved file
        os.remove(fname) 

cv.destroyAllWindows()

#print("{} patterns captured for calibration. Should be atleast 10.".format(patternsCaptured))

assert patternsCaptured >= 10, "Only {} patterns captured for calibration. Should be atleast 10.".format(patternsCaptured)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

calibrationDict = {
    'distortionCoefficients': dist.tolist(),
    'intrinsicMatrix': mtx.tolist()
}

#print(json.dumps(calibrationDict, indent=4))

if(os.name == "Windows"):
    jsonFileName = 'WinCamCalibration.json'
else:
    jsonFileName = 'MacCamCalibration.json'

print(os.name)

#Save calibrated cam params to json file from https://stackoverflow.com/a/26057360/13046931 
with open(jsonFileName, 'w') as fp:
    json.dump(calibrationDict, fp)

#Load in params from json file from https://www.geeksforgeeks.org/read-json-file-using-python/ 
with open(jsonFileName, 'r') as f:
    # returns JSON object as a dictionary
    returnedCalibrationDict = json.load(f)
    
# Iterating through the json list
for i in returnedCalibrationDict:
    print(i)

#convert retd lists back to numpy arrs
retdMatrix = np.asarray( returnedCalibrationDict['intrinsicMatrix'] )
retdCoeffs = np.asarray( returnedCalibrationDict['distortionCoefficients'] )

#Undistort another image from same cam *with json file params*

img = cv.imread('DistortedImage.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(retdMatrix, retdCoeffs, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, retdMatrix, retdCoeffs, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('UndistortedImage.png', dst)