#!/usr/bin/env python3
import os
import cv2 as cv 
print(cv.__version__)

#READ IN AND SHOW IMAGE
"""
#read in image
img = cv.imread('/Users/sethhi2/Downloads/IMG_1448.jpg')
key = ord('r')
while key != ord('s'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()
"""
#GET A WEB CAM CAPTURE
# already knows first webcam on mach to use

#grab device 0
"""
webcam = cv.VideoCapture(0)

while key != ord('s'):
    still = webcam.read()
    print(still)
    cv.imshow("OpenCV Webcam", still[1]) #first val is data type
    key = cv.waitKey(10) #lower wait key val is laggier + less fps
cv.destroyAllWindows()
"""
#read in image
img = cv.imread('/Users/sethhi2/Downloads/Seth Professional Photo.png')
# col (x), row (y), channel (z)
#BGR Color Space
# can access pixel using arr notation
print(img[40, 40, 0]) #prints out intensity
print(img.shape) #may use this to do some math, image dimensions
print(img.dtype)

#can index image
img = img[250:500, 400:700]
key = ord('r')
while key != ord('s'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()

#padding an image by addding border
# top, bot, left, right (border width)
border_size = 10
#color if border is const
border_color = [255, 0, 0]
# copy border pixels on edges to make new border image (REPLICATE = extending picels at end of border) ( or wrap to get pixels from other edge)
img = cv.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value = border_color) #border reflect (or reflect 101) (mirror pixels in)
key = ord('r')
while key != ord('s'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()

#WRITE OUT AN IMAGE TO SAVE IT (can specify image type)
path = os.getcwd()
cv.imwrite(path, img) 

#WRITE TEXT ON IMAGE
# to classify something
# for debugging if motion detected
def write(img, text, origin = (50,50), font = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,0,0), thickness = 2, line_type = cv.LINE_AA, botLeftOrigin = (0,0)):
    newImg = cv.putText(img, text, origin, font, fontScale, color, thickness, line_type, botLeftOrigin )
    return newImg    

write(img, "This is a picture")
key = ord('r')
while key != ord('s'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()