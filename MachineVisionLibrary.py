#!/usr/bin/env python3
import cv2 as cv 

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
    
#WRITE TEXT ON IMAGE
# to classify something
# for debugging if motion detected
def writeText(img, text, origin = (50,50), font = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,0,0), thickness = 2, line_type = cv.LINE_AA):
    newImg = cv.putText(img, text, origin, font, fontScale, color, thickness, line_type)
    return newImg

def saveImage(img, newPicName = "newPic", extension = ".jpg" ):
    #WRITE OUT AN IMAGE TO SAVE IT (need to specify image type)
    cv.imwrite(newPicName, img + extension) 