#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys

#used https://stackoverflow.com/a/4383597/13046931 to import lib
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '') #include src of repo at runtime
from MachineVisionLibrary import *

img = cv.imread("MeAndEmoly.jpg")
rows, cols, channels = img.shape

imgBlur = cv.blur(img, (7,7) ) #blurry bc big kernel
showImage(imgBlur)

imgGaussBlur = cv.GaussianBlur(img, (7,7), 0) # 0 to calc gaussian blur for us
showImage(imgGaussBlur)

imgMedBlur = cv.medianBlur(img, 9)
showImage(imgMedBlur)

imgEdgePreserveBlur = cv.bilateralFilter(img, 11, 61, 39)
showImage(imgEdgePreserveBlur)

#Conv to grey scale
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
showImage(imgGrey)
#pre proccing to reduce noise
imgGaussBlur = cv.GaussianBlur(imgGrey, (15,15), 0)
#should find vert lines bc dx (find places of big change)
#imgSobel = cv.Sobel(imgGaussBlur, -1, 1, 0, ksize=5) #image, depth, order of x deriv, order of y deriv, kernel size
#should find horiz lines bc dy
#imgSobel = cv.Sobel(imgGaussBlur, -1, 0, 0, ksize=5)
imgSobel = cv.Sobel(imgGaussBlur, -1, 1, 1, ksize=5)
showImage(imgSobel)

