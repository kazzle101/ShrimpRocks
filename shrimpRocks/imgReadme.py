#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os
import sys

from shrimpRocks.imgUtilities import ImageUtilities
from shrimpRocks.imgFilters import ImageFilters
from shrimpRocks.samProcess import SAMprocess
from shrimpRocks.imgAnalyse import ImageAnalyse
from shrimpRocks.imgCropping import ImageCropping

""" Generate image files for use in the README.md file to illustrate the filtering steps."""
class ImageReadme():
    
    def __init__(self, oneCentimeter, sourceDir):
        self.sourceDir = sourceDir
        self.windowTitle = "Readme Images"
        self.oneCentimeter = oneCentimeter
        return
        
    def appendFilter(self, filterList: list, filter: str, offset: int=4) -> tuple:
        
        filterList.append(filter)
        return len(filterList)+offset, filterList
    
    def makeReadmeImages(self, imgID:int, image_file: str, output_dir: str):
        
        imageUtils = ImageUtilities()        
        imageAnalyse = ImageAnalyse(self.oneCentimeter, output_dir)
        imageCropping = ImageCropping(output_dir)
        imageFilters = ImageFilters()
        samProc = SAMprocess()        
                
        print(f"Loading image and generating SAM masks for {image_file} (One-time cost)...")
        # 1. Initialization (Run SAM only once)
        mask_generator = samProc.load_sam()
        image, image_rgb = samProc.load_image(image_file)
        sam_masks_data = samProc.generate_masks(mask_generator, image_rgb)
        
        sourceFile = os.path.join(self.sourceDir,f"Still 2024-09-20 230424_1.2.{imgID}.png")
        try:
            source_image = cv2.imread(sourceFile)            
        except Exception as e:
            print(f"CV2 Cannot load image: {sourceFile}")
            print(e)
            sys.exit()
                     
        imageUtils.saveImage(os.path.join(output_dir,"01_source_image.png"), source_image)
        _, testImg = imageCropping.selectInsideYellowSquareImage(source_image)
        imageUtils.saveImage(os.path.join(output_dir,"02_rulers_selected.png"), testImg)        
        imageUtils.saveImage(os.path.join(output_dir,"03_source_pebbles.png"), image)

        output_image = samProc.makeOutputImage(image, sam_masks_data)
        imageUtils.saveImage(os.path.join(output_dir,"04_all_pebbles_selected.png"), output_image)
                
        thesefilters = []        
        fid, thesefilters = self.appendFilter(thesefilters, "minimumSize")     
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_minimum_size.png"), output_image)
        
        fid, thesefilters = self.appendFilter(thesefilters, "touchingEdges")
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_touching_edge.png"), output_image)
        
        fid, thesefilters = self.appendFilter(thesefilters, "occluded")        
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_occluded.png"), output_image)
        
        fid, thesefilters = self.appendFilter(thesefilters, "wholeness")        
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_wholeness.png"), output_image)
            
        # fid, thesefilters = self.appendFilter(thesefilters, "convexity")        
        # filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        # output_image = samProc.makeOutputImage(image, filtered_masks)
        # imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_convexity.png"), output_image)            
        
        fid, thesefilters = self.appendFilter(thesefilters, "complexity")        
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_complexity.png"), output_image)
        
        fid, thesefilters = self.appendFilter(thesefilters, "roundish")        
        filtered_masks, pebble_data = imageFilters.applyfilters(image, sam_masks_data, thesefilters)        
        output_image = samProc.makeOutputImage(image, filtered_masks)
        imageUtils.saveImage(os.path.join(output_dir,f"{fid:02d}_filter_roundish.png"), output_image)

        total_pebbles, average_size, average_wholeness = imageAnalyse.calculate_average_size_and_wholeness(pebble_data)

        # Display Results
        print(f"files saved to: {output_dir}")
        print(f"Image analyised: {image_file}")
        print(f"Total pebbles counted: {total_pebbles}")
        print(f"Average size of measured pebbles: {average_size:.2f} pixels, {imageAnalyse.pxAreaToCM2(average_size):.2f} cm^2")        
        return        
        