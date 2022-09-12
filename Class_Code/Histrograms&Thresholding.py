import cv2 as cv
from MachineVisionLibrary import mv_functs
import matplotlib.pyplot as plt

img = cv.imread('MeAndEmoly.jpg')

rows, cols, channels = img.shape

imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mv_functs.showImage(imgGrey)

#BRIGHTNESS HISTOGRAM
histogram = cv.calcHist([imgGrey], [0], None, [256], [0, 256])

plt.figure()
plt.title("Luminosity Histogram")
plt.xlabel("bins")
plt.ylabel("Number of pixels in each bin")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()

#easier to see background or dark areas
imgEqualized = cv.equalizeHist(imgGrey)

mv_functs.showImage(imgEqualized)

#THRESHOLD

#BINARY
#ret, imgThresh = cv.threshold(imgGrey, 127, 255, cv.THRESH_BINARY) #assumes 0 is lowest
ret, imgThreshBin = cv.threshold(imgGrey, 80, 255, cv.THRESH_BINARY)
mv_functs.showImage(imgThreshBin)

#INV BINARY
ret, imgThreshInvBin = cv.threshold(imgGrey, 127, 255, cv.THRESH_BINARY_INV)
mv_functs.showImage(imgThreshInvBin)

#TRUNC BINARY
ret, imgThreshTrunc = cv.threshold(imgGrey, 127, 255, cv.THRESH_TRUNC)
mv_functs.showImage(imgThreshTrunc) 

