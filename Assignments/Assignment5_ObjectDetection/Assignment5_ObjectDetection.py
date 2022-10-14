#!/usr/bin/env python3
"""
Author: Seth Cram
Class: CS404 - Machine and Robotic Vision
Assignment 4 - Motion Detector
"""

from cv2 import FlannBasedMatcher_create
import numpy as np
import cv2 as cv
from MachineVisionLibrary import mv_functs
import matplotlib.pyplot as plt

class Img:
    """
    Class to store data associated with an image.
    """
    def __init__(self, img) -> None:
        self.img = img
        self.kp = []
        self.descr = []

#init option strs

#feature detection
ORB_STR = 'ORB'
SIFT_STR = 'SIFT'
SURF_STR = 'SURF'
FAST_STR = 'FAST'
BRIEF_STR = 'BRIEF'
BLOB_STR = 'BLOB'
BRISK_STR = 'BRISK'
#feature matching 
BRUTE_FORCE_STR = 'BruteForce'
FLANN_STR = 'FLANN'

class ObjectDetection:
    """
    Class to use a single feature detection method 
    with a single feature matching method
    """
    
    def __init__(self, featureDetection, featureMatching) -> None:
        self.featureDetection = featureDetection
        self.featureMatching = featureMatching
        
    def ResizeImage(self, img, resizeBy):
        return cv.resize( img, (int(img.shape[1]*resizeBy), int(img.shape[0]*resizeBy)) )
    
    def MatchOps(self, matchingObj, img1, img2, descr1, descr2, kp1, kp2):
        
        if len(descr1) == 0:
            raise ValueError("MatchFinder 1st descriptor empty")   
        if len(descr2) == 0:
            raise ValueError("MatchFinder 2nd descriptor empty")
        
        #find the matches using KNN of matching obj
        matches = matchingObj.knnMatch(descr1, descr2, k=2)
        
        #if BF matcher
        if self.featureMatching == BRUTE_FORCE_STR:
            #match descriptions
            matches = matchingObj.match(descr1, descr2)
            
            #sort matches by distance
            matches = sorted(matches, key=lambda x:x.distance)
            #Draw first 20 closest matches (could also use KNN)
            return cv.drawMatches(img1, kp1, img2, kp2, matches[0:20], flags=2, outImg=None)
            
        #If FLANN matcher
        elif self.featureMatching == FLANN_STR:
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
            
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
            
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        else:
            raise ValueError(f"The feature matching method {self.featureMatching} is not implemented.")
    
    def run(self, resizeBy: float = 0.1) -> None:
        #read in greyscaled 3 images of same obj in diff pos's/orientations
        img1 = cv.imread("img1.jpg", cv.IMREAD_GRAYSCALE)
        img2 = cv.imread("img2.jpg", cv.IMREAD_GRAYSCALE)
        img3 = cv.imread("img3.jpg", cv.IMREAD_GRAYSCALE)
        
        #cache imgs as arr + calc len
        imgsArr = np.array([Img(img1), Img(img2), Img(img3)])
        imgsArrLen = len(imgsArr)
        
        #resize images
        for i in range(imgsArrLen):
            imgsArr[i].img = self.ResizeImage(imgsArr[i].img, resizeBy)
        
        #decide identifying method
        if self.featureDetection == ORB_STR:
            detectionObj = cv.ORB_create()
        elif self.featureDetection == SIFT_STR:
            detectionObj = cv.SIFT_create()
        elif self.featureDetection == BRISK_STR:
            detectionObj = cv.BRISK_create()
       
        #elif self.featureDetection == SURF_STR:
        #    minHessian = 400
        #    detectionObj = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)

            #detectionObj = cv.SimpleBlobDetector_create()
            
        else:
            raise ValueError(f"The feature detection method {self.featureDetection} is not implemented.")
        
        #identify + descr features
        for i in range(imgsArrLen):
            imgsArr[i].kp, imgsArr[i].descr = detectionObj.detectAndCompute(imgsArr[i].img, None)
        
        #decide matching method
        if self.featureMatching == BRUTE_FORCE_STR:
            #assign normalizaton type of BF matcher based on feature method
            if self.featureDetection in [ORB_STR, BRISK_STR, FAST_STR, BRIEF_STR]:
                normType = cv.NORM_HAMMING
            #for sift and surf (not sure what to do for BLOB)
            elif self.featureDetection in [SIFT_STR, SURF_STR]:
                normType = cv.NORM_L2
            else:
                raise ValueError(f"Don't know how to use brute force matching with the feature detection method {self.featureDetection}.")
        
                
            #create matching obj w/ desired norm type
            matchingObj = cv.BFMatcher_create(normType)
        elif self.featureMatching == FLANN_STR:
            #create flann descriptor matcher
            #matchingObj = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            
            #if float feature detection
            if self.featureDetection in [SIFT_STR, SURF_STR]:
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)   # or pass empty dictionary
            #if binary feature detection
            elif self.featureDetection in [ORB_STR, BRISK_STR]:
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
                search_params = dict()
            else:
                raise ValueError(f"Don't know how to use flann matching with the feature detection method {self.featureDetection}.")
                
            matchingObj = cv.FlannBasedMatcher(index_params, search_params)
            
        for j in range(1, imgsArrLen):
            #perform the match ops w/ img0 as base
            imgsArr[j].img = self.MatchOps(
                matchingObj, 
                imgsArr[0].img , 
                imgsArr[j].img , 
                imgsArr[0].descr, 
                imgsArr[j].descr, 
                imgsArr[0].kp, 
                imgsArr[j].kp
            )
            
            #display image w/ matches draw on 
            mv_functs.showImage(imgsArr[j].img)

if __name__ == "__main__":
    
    resizeImagesBy = 0.1

    odOrbBruteforce = ObjectDetection(featureDetection=ORB_STR, featureMatching=BRUTE_FORCE_STR)
    odOrbBruteforce.run(resizeImagesBy)

    odSIFTBruteforce = ObjectDetection(featureDetection=SIFT_STR, featureMatching=BRUTE_FORCE_STR)
    odSIFTBruteforce.run(resizeImagesBy)

    odSIFTBruteforce = ObjectDetection(featureDetection=BRISK_STR, featureMatching=BRUTE_FORCE_STR)
    odSIFTBruteforce.run(resizeImagesBy)
    
    odSIFTBruteforce = ObjectDetection(featureDetection=ORB_STR, featureMatching=FLANN_STR)
    odSIFTBruteforce.run(resizeImagesBy)
    
    odSIFTBruteforce = ObjectDetection(featureDetection=SIFT_STR, featureMatching=FLANN_STR)
    odSIFTBruteforce.run(resizeImagesBy)
    
    odSIFTBruteforce = ObjectDetection(featureDetection=BRISK_STR, featureMatching=FLANN_STR)
    odSIFTBruteforce.run(resizeImagesBy)