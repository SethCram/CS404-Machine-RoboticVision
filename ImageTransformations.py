from random import betavariate
import numpy as np
import cv2 as cv
from MachineVisionLibrary import *

img = cv.imread('MeAndEmoly.jpg')
rows, cols, channels = img.shape

#Transformation
M = np.float32([[1, 0, 100], [0, 1, 50]])
#M 
# 1 0 100
# 0 1 50
imgTranslated = cv.warpAffine(img, M, (cols, rows))
showImage(imgTranslated)

#Rotation
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
imgRotated = cv.warpAffine(img, M, (cols, rows))
showImage(imgRotated)

#RESIZE/SCALING (common w/ passing to AI models)

#enlarge
#imgResized = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_NEAREST )
#imgResized = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_LINEAR )
#imgResized = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_CUBIC )
imgResized = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_AREA )

#shrink
#imgResized = cv.resize(img.copy(), None, fx=0.1, fy=0.1, interpolation=cv.INTER_CUBIC)

showImage(imgResized)

#General Affine Tranformation (useful for general rot)
first_pnts = np.float32([ [50,50], [200,50], [50,200] ])
nxt_pnts = np.float32([ [10,100], [200,50], [100,250] ])
M = cv.getAffineTransform(first_pnts, nxt_pnts) #openCV figure out transform for u if dont want specific transform
imgRotated = cv.warpAffine(img, M, (cols, rows)) 
showImage(imgRotated)

#Perspective Transform
first_pnts = np.float32([ [50,50], [400,50], [50,400], [400,400] ]) #uses 4 bc all 4 corners?
nxt_pnts = np.float32([ [50,0], [600,0], [50,50], [600,600] ])
M = cv.getPerspectiveTransform(first_pnts, nxt_pnts)
imgPerspective = cv.warpPerspective(img, M, (cols, rows))
showImage(imgPerspective)

#COLOR CHANNELS
b, g, r = cv.split(img)
showImage(b)
showImage(g)
showImage(r)

# (possible not working properly) (if working properly only supposed to show 1 pixel??)
blue = img[:, :, 2] = 0 #numpy faster (zeros out index 2 so no red?)
showImage(blue)

#Brightness/Contrast
#output_pixel = (input_pixel * alpha ) + beta
#alpha = constrast/gain
#beta = bias/brightness
alpha = 2.2
beta = 20.0

imgBright = np.empty(img.shape)

#transform pixel by pixel (takes long time) (brightens picture)
for row_pixel in range(0, img.shape[0]):
    for col_pixel in range(0, img.shape[1]):
        for channel_pixel in range(0, img.shape[2]):
            imgBright[row_pixel][col_pixel][channel_pixel] = np.clip( img[row_pixel][col_pixel][channel_pixel]*alpha + beta, 0, 255 )
#if no clipping, vals fall below 0 and above 255

showImage(imgBright)

#brighten using funct
imgBright = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
showImage(imgBright)

#Gamma Correction
#gamma = 1.0
#gamma = 0.1
gamma = 5
look_up = np.empty((1, 256), np.uint8)
for  i in range(256):
    look_up[0, i] = np.clip(pow(i/255.0, gamma)*255.0, 0, 255) 
imgGammafied = cv.LUT(img, look_up)

showImage(imgGammafied)