
from __future__ import print_function
import cv2 as cv
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1,im2):
    im1Gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1,descriptors1 = orb.detectAndCompute(im1Gray,None)
    keypoints2,descriptors2 = orb.detectAndCompute(im2Gray,None)

    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1,descriptors2,None)
    matches=sorted(matches,key=lambda x:x.distance,reverse=False)


    numGoodMatches = int(len(matches)*GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    imMatches = cv.drawMatches(im1,keypoints1,im2,keypoints2,matches,None)
    cv.imwrite("matches.jpg",imMatches)

    points1 = np.zeros((len(matches),2),dtype=np.float32)
    points2 = np.zeros((len(matches),2),dtype=np.float32)
    for i,match in enumerate(matches):
        points1[1,:] = keypoints1[match.queryIdx].pt
        points2[1,:] = keypoints2[match.trainIdx].pt

    
    h,mask = cv.findHomography(points1,points2,cv.RANSAC)

    height,width,channels = im2.shape
    im1REG = cv.warpPerspective(im1, h, (width, height))

    return im1REG , h
if __name__ =="__main__":
    refFilename = "quaternion_rifle1.jpg"

    print("Reading reference image : ", refFilename)

    imReference = cv.imread(refFilename, cv.IMREAD_COLOR)


    imFilename = "quaternion_helmet1.jpg"

    print("Reading image to align : ", imFilename)

    im = cv.imread(imFilename, cv.IMREAD_COLOR)

    print("Aligning images ...")
    imReg, h = alignImages(im, imReference)

    outFilename = "aligned.jpg"

    print("Saving aligned image : ", outFilename)

    cv.imwrite(outFilename, imReg)

    print("Estimated homography : \n",  h)
    





