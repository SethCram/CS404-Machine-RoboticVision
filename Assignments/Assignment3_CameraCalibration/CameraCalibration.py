import numpy as np
import cv2 as cv
import glob
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '') #include src of repo at runtime

from MachineVisionLibrary import *

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#GET WEB CAM CAPTURES

#grab device 0 (TAKES A LONG TIME) (works on Windows+Mac)
webcam = cv.VideoCapture(0)

images = []
saveImage = True

#want pattern board to be in various angles for this

#show webcam footage
while (True):
    still = webcam.read()
    print(still)
    cv.imshow("OpenCV Webcam", still[1]) #first val is data type
    
    if( saveImage ):
        decision = input("Would you like to save this still? (y/n)")
        
        if(decision == "y"):
            saveImage(still[1])
        elif(decision == "n"):
            print("Image not saved")
    
    #add curr still to images
    images.append(still[1])
    
    # desired button of your choice
    if cv.waitKey(10) & 0xFF == ord('c'):
        break
    
# After the loop release the cap object
webcam.release()
cv.destroyAllWindows()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob('*.jpg')

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
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Save calibrated cam params to json file

#Load in params from json file
# undistort another image from same cam with them

#Credit tutorial code