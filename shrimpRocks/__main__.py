#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Chesil Beach Pebble Survey Part 1 - Spending the Whole Day Looking at Gravel
# https://www.youtube.com/watch?v=8RAuvyWM_2E

# sudo apt install python3-opencv
# sudo apt install python3-natsort
# sudo apt install python3-scipy
# sudo apt install python3-sklearn  # scikit-learn
# sudo apt install python3-skimage  # scikit-image
# sudo pip install opencv-contrib-python --break-system-packages

import argparse
import sys
import os

_imageDir = "images/"
_sourceDir = os.path.join(_imageDir, "source/")
_imageCroppedDir = os.path.join(_imageDir, "cropped/")
_imageAnalysedDir = os.path.join(_imageDir, "analysed/")
_imageTestDir = os.path.join(_imageDir, "test/")
# _settingsFile = os.path.join(_imageDir, "shrimpsettings.json")
_oneCentimeter = 75  # pixels

from shrimpRocks.getFiles import GetFiles
from shrimpRocks.imgUtilities import ImageUtilities
from shrimpRocks.imgCropping import ImageCropping
from shrimpRocks.clkImage import ClickImage
from shrimpRocks.imgAnalyse import ImageAnalyse
from shrimpRocks.imgReadme import ImageReadme

def main():
    
    getfiles = GetFiles()
    imgUtils = ImageUtilities()
    imgCropping = ImageCropping(_imageDir)
    imgAnalyse = ImageAnalyse(_oneCentimeter)
    clkImage = ClickImage(_oneCentimeter)
    
    desc = f"""Futility for measuring pebble sizes on Chesil Beach.\n
    Image numbers are in the range 1 to 33 and correspond to those found in the {_sourceDir} or {_imageCroppedDir} directories"""
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-p","--process", action='store_true', help="Crop the original images ready for examination")
    parser.add_argument('-a', '--averagesize', action='store_true', help='Show the average pebble sizes and output a plot')  
    parser.add_argument('--croptest', type=int, help='Use the image number to test an individual image, for checking the crop process is working', default=0)
    parser.add_argument('--segment', type=int, help='Filer Test, using the image number display an indivdual rock image with the filters applied')
    parser.add_argument('--chug', type=int, default=None, help=f'Filter Test, use an image number for testing a filter with a range of values, files are output to {_imageTestDir}')      
    parser.add_argument('--makereadme', type=int, default=None, help=f'Make images for the readme.md file using an image number')
    parser.add_argument('--clickimage', type=int, default=None, help=f'Using an image number, loads a filtered image, allows you to click on the masks for information about the mask')

    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return   
    
    if args.averagesize:        
        images = getfiles.filesList(_imageCroppedDir)
        if images is None:
            print (f"no cropped images found in: {_imageCroppedDir}")
            return
            
        getfiles.makeOutputDir(_imageAnalysedDir)
        getfiles.deleteFiles(_imageAnalysedDir)
        sizes = imgAnalyse.makeAverageSizes(images, _imageAnalysedDir)
        imgAnalyse.plotAverageSizes(sizes, _imageDir)
        return
    
    if args.clickimage:
        imgID = args.clickimage
        images = getfiles.filesList(_imageCroppedDir)       
        filename = getfiles.isRockfordFile(images,imgID)
        if filename is None:
            print(f"Image {imgID} not found")
            return
        
        clkImage.makeClickImage(filename)
        return
    
    if args.segment:
        imgID = args.segment 
        images = getfiles.filesList(_imageCroppedDir)       
        filename = getfiles.isRockfordFile(images,imgID)
        if filename is None:
            print(f"Image {imgID} not found")
            return
        
        imgAnalyse.runSegment(filename)
        return
            
    if args.croptest:        
        imgID = args.croptest
        images = getfiles.filesList(_sourceDir)
        filename = getfiles.isRockfordFile(images, imgID)  
        if filename is None:
            print(f"Image {imgID} not found")
            return        
                          
        img, testImg = imgCropping.selectInsideYellowSquare(filename, True)    
        output_image = imgUtils.concat_same_height(testImg, img)
        print ("On the left, showing the verticals and horizontals selected in red and green, and right, the cropped area")
        print ("press any key while inside the image to exit")
        imgUtils.showImage(output_image)
        return
    
    if args.process:
        images = getfiles.filesList(_sourceDir)
        if images is None:
            print (f"no cropped images found in: {_imageCroppedDir}")
            return        
        
        getfiles.makeOutputDir(_imageCroppedDir)
        getfiles.deleteFiles(_imageCroppedDir)
        
        print("looking for pebbles")        
        c=1
        for i in images:
            img, _ = imgCropping.selectInsideYellowSquare(i)
            filename = os.path.join(_imageCroppedDir,f"rocks_{str(c).zfill(2)}.png")      
            imgUtils.saveImage(filename, img)
            c+=1
    
        print("done")
        return
        
    if args.makereadme:
        imgID = args.makereadme    
        images = getfiles.filesList(_imageCroppedDir)
        filename = getfiles.isRockfordFile(images, imgID)
        if filename is None:
            print(f"Image {imgID} not found, or file {filename} not found")
            return
                
        imgReadme = ImageReadme(_oneCentimeter, _sourceDir)
        output_dir = os.path.join(_imageDir, "readmeImgs/")
        getfiles.makeOutputDir(output_dir)
        getfiles.deleteFiles(output_dir)

        imgReadme.makeReadmeImages(imgID, filename, output_dir) 
        return
        
    if args.chug:
        imgID = args.chug    
        images = getfiles.filesList(_imageCroppedDir)
        filename = getfiles.isRockfordFile(images,imgID)        
        if filename is None:
            print(f"Image {imgID} not found, or file {filename} not found")
            return        
        
        chugTestDir = os.path.join(_imageDir, "chugtest/")
        getfiles.makeOutputDir(chugTestDir)
        getfiles.deleteFiles(chugTestDir)          
        imgAnalyse.chugSegment(filename, imgID, chugTestDir)
        return

    return

if __name__ == "__main__":
    main()
