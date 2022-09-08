import numpy as np
import cv2 as cv
import glob
import sys
import json

#used https://stackoverflow.com/a/4383597/13046931 to import lib
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '') #include src of repo at runtime
from MachineVisionLibrary import *

decision = input("Do you need to take more pictures for calibration?(y/n)")

#if needa take more pics
if(decision == "y"):

    #GET WEB CAM CAPTURES from https://stackoverflow.com/a/34588758/13046931

    #grab device 0 (TAKES A LONG TIME) (works on Windows+Mac)
    webcam = cv.VideoCapture(0)

    #images = []

    #want pattern board to be in various angles for this
    # displays the camera feed in a cv2.namedWindow and will take a snapshot when you hit SPACE. It will also quit if you hit ESC.

    cv.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("test", frame)

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
            
            #STORE IMAGE FOR CALIBRATION
            #add curr still to images
            #images.append(frame)
            #img_name = "opencv_frame_{}.png".format(img_counter)
            #print("{} saved for calibration!".format(img_name))
            #img_counter += 1
        
    webcam.release()
    cv.destroyAllWindows()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

patternsCaptured = 0
calibrationDict = { }

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

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
        patternsCaptured += 1

cv.destroyAllWindows()

print("{} patterns captured for calibration. Should be atleast 10.".format(patternsCaptured))

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#save intrinsic matrix parameters and the distortion coefficients 
calibrationDict[mtx] = dist

jsonFileName = 'CamCalibration.json'

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

#Undistort another image from same cam *with json file params*

img = cv.imread('DistortedImage.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('UndistortedImage.png', dst)