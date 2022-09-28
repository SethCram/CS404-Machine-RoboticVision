import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs

if __name__ == "__main__":
    #open webcam
    webcam = cv.VideoCapture(0)

    key = ord('r')

    cv.namedWindow(mv_functs.Impl_Consts.CONTROLS_PANEL_NAME)
        
    cv.createTrackbar("blur kernel", "controls", 3, 100, mv_functs.nothing)
    cv.setTrackbarPos('blur kernel', 'controls', 3)
    
    #create bg subtractor
    bg = cv.createBackgroundSubtractorKNN() 
    
    while (True):
        
        still = webcam.read()
        
        og_img = still[1]
        
        #should undistortion json info be used? or just for images and not webcame caps?
            
        #conv to greyscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        
        blurLevel = int( cv.getTrackbarPos('blur kernel', mv_functs.Impl_Consts.CONTROLS_PANEL_NAME))
        #turn even numbers odd bc odd kernel needed
        if blurLevel % 2 == 0:
            blurLevel += 1
        
        #gaussian blur
        img = cv.GaussianBlur(img, (blurLevel, blurLevel), 0)
        
        #subtr background 
        img = bg.apply(img)
        
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        contours = list(contours)
        contours.sort(key=cv.contourArea, reverse=True)
        # throws "list index out of range" if blur kernel set too high (22 max?) 
        #  or if bg subtractor used before image bounding
        
        if(contours is not None and len(contours) > 0):
            contours = contours[0] 
        else:
            continue
        
        #cv.drawContours(og_img, contours, -1, (255, 0, 0), 3)
        #img = og_img
        
        #bounding rect
        x,y,w,h = cv.boundingRect(contours)
        cv.rectangle(og_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        #rot'd rect (more conservative w/ area and rots to fit shape)
        #rect = cv.minAreaRect(contours)
        #box = cv.boxPoints(rect)
        #box = np.int0(box)
        #cv.drawContours(og_img, [box], 0, (0,0,255), 2)
        
        #min enclosing circ
        #(x,y), radius = cv.minEnclosingCircle(contours)
        #center = (int(x), int(y))
        #radius = int(radius)
        #cv.circle(og_img, center, radius, (0,255,0), 2)
        
        cv.imshow(mv_functs.Impl_Consts.IMAGE_WINDOW_NAME, og_img) #first val is data type
        
        if cv.waitKey(5) & 0xFF == ord(mv_functs.Impl_Consts.CLOSE_KEY):
            break
        
    # After the loop release the cap object
    webcam.release()
    cv.destroyAllWindows()