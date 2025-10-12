#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os
import sys

from shrimpRocks.imgUtilities import ImageUtilities

class ImageCropping():
    
    def __init__(self, testDir="images/"):
        self.testDir = testDir
        ## allowance angle for the horzontal and vertical lines that are in the images
        self.angleTolerance = 12 
        ## minimum distance in pixels between the left/top inside edge of the ruler 
        # and the right/bottom inside edge of the ruler 
        self.minDistance = 1000
        ## padding in pixels when cropping the top and left of the images, so as to remove
        # any remains of the ruler and notebook
        self.cropPadding = 50        
        ## after the top and left have been cropped, crop the image to be square and remove
        # the ruler on the right and bottom
        self.cropSquare = 1200
        
        ## used by cv2.HoughLinesP,
        # see: https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
        self.rho=2 
        self.threshold=100
        self.minLineLength=400
        self.maxLineGap=5
        
        return
    
    def selectInsideYellowSquare(self, imagePath: str, testMode: bool=False) -> tuple:

        try:
            image = cv2.imread(imagePath)
        except Exception as e:
            print(f"CV2 Cannot load image: {imagePath}")
            print(e)
            sys.exit()
        
        image, testImage = self.selectInsideYellowSquareImage(image)                
        return image, testImage
    
    def selectInsideYellowSquareImage(self, image: list, testMode: bool=False) -> tuple:
        # imgUtils = ImageUtilities()
    
        # crop the right hand side, where the GPS information is.
        image = image[:, 0:1980]
       
        # find the inner top and left of the square marked out by the ruler
        x, y, testimg = self.detectTopAndLeftInsideEdges(image)        
        if x is None or y is None:
            return None
        
        # crop the left and top plus some padding
        image = image[y+self.cropPadding:, x+self.cropPadding:]
                        
        # finally crop the image square, to remove the ruler from the top and bottom
        image = image[:self.cropSquare, :self.cropSquare]
                                                                                
        return image, testimg
    
    def detectTopAndLeftInsideEdges(self, image: list) -> tuple:
        
        testImg = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)#
        
        # Use Hough Line Transform to detect straight lines 
        lines = cv2.HoughLinesP(edges, rho=self.rho, theta=np.pi/180, threshold=self.threshold, 
                                minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        
        # Find the positions of the mostly horizontal and vertical lines
        verticalLines = []
        horizontalLines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if self.isMostlyVertical(x1, y1, x2, y2):
                x_avg = (x1 + x2) // 2                
                cv2.line(testImg,(x1,y1),(x2,y2),(0,0,255),2)                
                verticalLines.append(x_avg)  
        
            if self.isMostlyHorizontal(x1, y1, x2, y2):
                y_avg = (y1 + y2) // 2                
                cv2.line(testImg,(x1,y1),(x2,y2),(0,255,0),2)                
                horizontalLines.append(y_avg)  

        if len(verticalLines) == 0:
            print ("no vertical lines found")
            return [None, None]
        
        if len(horizontalLines) == 0:
            print ("no horizontal lines found")
            return [None, None]        
        
        # Sort lines by their coordinates
        verticalLines = sorted(verticalLines) 
        horizontalLines = sorted(horizontalLines)
        
        # Find the first vertical and horizontal line where the next is more than minDistance pixels away
        xEdge = None
        yEdge = None
        for i in range(len(verticalLines) - 1):
            diff = verticalLines[i + 1] - verticalLines[i]
            #print (f"X: {verticalLines[i + 1]} - {verticalLines[i]} = {diff}")            
            if diff > self.minDistance:                
                xEdge = verticalLines[i]
                break
            
        for i in range(len(horizontalLines) - 1):
            diff = horizontalLines[i + 1] - horizontalLines[i]
            #print (f"Y: {horizontalLines[i + 1]} - {horizontalLines[i]} = {diff}")            
            if diff > self.minDistance:                
                yEdge = horizontalLines[i]
                break
        
        return xEdge, yEdge, testImg
    
    
    # Vertical lines should have angles close to 90 degrees or -90 degrees 
    def isMostlyVertical(self, x1: int, y1: int, x2: int, y2: int) -> bool:        
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)
        isVertical = abs(angle - 90) <= self.angleTolerance
        #print(f"{angle:.2f},\t {isVertical} xy1:{x1},{y1}, xy2:{x2},{y2}")
        return isVertical
    
    # Horizontal lines have angles close to 0 or 180 degrees.
    def isMostlyHorizontal(self, x1: int, y1: int, x2: int, y2: int) -> bool:        
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)
        isHorizontal = (abs(angle - 0) <= self.angleTolerance or abs(angle - 180) <= self.angleTolerance)
        #print(f"{angle:.2f},\t {isHorizontal} xy1:{x1},{y1}, xy2:{x2},{y2}")
        return isHorizontal
    
    