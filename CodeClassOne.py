#!/usr/bin/env python3
import cv2 as cv 

CLOSE_KEY = 'c'

#read in image
img = cv.imread('MeAndEmoly.jpg')

def showImage(image, windowName = "OpenCV Picture", closeKey = 'c'):
    """Shows the given image until the closeKey is pressed.

    Args:
        image (_type_): Image to show.
        windowName (str, optional): Desired display window name. Defaults to "OpenCV Picture".
        closeKey (str, optional): Key that needs to be pressed to close window. Defaults to 'c'.
    """
    #read 
    key = ord('r')

    #loop till close key pressed
    while key != ord(closeKey):
        cv.imshow(windowName, image)
        key = cv.waitKey()
    
    #destroy windows
    cv.destroyAllWindows()
    

#READ IN AND SHOW IMAGE

#read in image
key = ord('r')

while key != ord('c'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()

#GET A WEB CAM CAPTURE
# already knows first webcam on mach to use

#grab device 0 (TAKES A LONG TIME) (works on Windows)
webcam = cv.VideoCapture(0)

#show webcam footage
while (True):
    still = webcam.read()
    print(still)
    cv.imshow("OpenCV Webcam", still[1]) #first val is data type
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(10) & 0xFF == ord('c'):
        break
    
# After the loop release the cap object
webcam.release()
cv.destroyAllWindows()

# col (x), row (y), channel (z)
#BGR Color Space
# can access pixel using arr notation
print(img[40, 40, 0]) #prints out intensity
print(img.shape) #may use this to do some math, image dimensions
print(img.dtype)

#can index image
img = img[250:500, 400:700]
key = ord('r')
while key != ord('c'):
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
while key != ord('c'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()

#WRITE OUT AN IMAGE TO SAVE IT (need to specify image type)
newFileName = "ZoomedInWithBorder_MeAndEmoly.jpg"
cv.imwrite(newFileName, img) 

#WRITE TEXT ON IMAGE
# to classify something
# for debugging if motion detected
def writeText(img, text, origin = (50,50), font = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,0,0), thickness = 2, line_type = cv.LINE_AA):
    newImg = cv.putText(img, text, origin, font, fontScale, color, thickness, line_type)
    return newImg    

writeText(img, "This is a picture")
key = ord('r')
while key != ord('c'):
    cv.imshow("OpenCV Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()