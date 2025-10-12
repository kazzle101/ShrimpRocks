#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math 
import traceback
import matplotlib.pyplot as plt

from shrimpRocks.imgUtilities import ImageUtilities
from shrimpRocks.imgFilters import ImageFilters
from shrimpRocks.samProcess import SAMprocess

class ImageAnalyse():
    
    def __init__(self, oneCentimeter=75, outDir=None):
        self.oneCentimeter = oneCentimeter  # pixels 
        self.windowTitle = "Image Analyse"
        self.outDir = outDir
        return
    
    def calculate_average_size(self, areas: list) -> tuple:
        """Calculates the average size and returns the total count and average."""
        total_pebbles = len(areas)
        if total_pebbles == 0:
            return 0, 0
    
        total_area = sum(areas)
        average_area = total_area / total_pebbles    
        return total_pebbles, average_area
    
    def calculate_average_size_and_wholeness(self, pebble_data: list) -> tuple:
        """Calculates the average size and wholeness score."""
        total_pebbles = len(pebble_data)
        if total_pebbles == 0:
            return 0, 0, 0
    
        # Unpack the list of tuples into separate lists for calculation
        areas = [data[0] for data in pebble_data]
        solidity_scores = [data[1] for data in pebble_data]

        total_area = sum(areas)
        average_area = total_area / total_pebbles
        
        total_solidity = sum(solidity_scores)
        average_solidity = total_solidity / total_pebbles
    
        return total_pebbles, average_area, average_solidity
    
    def pxAreaToCM2(self, pxArea: float) -> float:
        """Converts pixel area to square centimeters."""
        cmArea = (pxArea / (self.oneCentimeter * self.oneCentimeter))        
        return cmArea
    
    def makeAverageSizes(self, image_list: list, imageAnalyseDir: str) -> list:
        samProc = SAMprocess() 
        imgFilters = ImageFilters()
        imageUtils = ImageUtilities() 
        
        # apply these filters
        filterList = ["minimumSize","touchingEdges","occluded", "wholeness", "convexHull", "complexity", "roundish"] #"convexHull",
        sizes = []        
        mask_generator = samProc.load_sam()
        
        id = 1
        for image_file in image_list:
            image, image_rgb = samProc.load_image(image_file)        
            sam_masks = samProc.generate_masks(mask_generator, image_rgb)
            
            filtered_masks, pebble_data = imgFilters.applyfilters(image, sam_masks, filterList=filterList)
            total_pebbles, average_size, _ = self.calculate_average_size_and_wholeness(pebble_data)
            cmArea = self.pxAreaToCM2(average_size)
            
            imgFile = os.path.basename(image_file)
            output_image = samProc.makeOutputImage(image, filtered_masks)
            imageUtils.saveImage(os.path.join(imageAnalyseDir,f"filtered_{imgFile}"), output_image)
            
            print(f"{imgFile}: {total_pebbles:03d} pebbles selected, Average Size: {average_size:.2f} pixels, {cmArea:.2f} cm^2")
            sizes.append({"id": id, "imageFile": imgFile, "pxArea": average_size, "cmArea": cmArea})
            id=id+1
        
        print(f"Filtered images saved to: {imageAnalyseDir}")
        # cv2.destroyAllWindows()
        return sizes
    
    def plotAverageSizes(self, sizes: list, plotOutDir: str):
        
        print("Displaying results, press any key while in the plot to exit")
        outPlot = os.path.join(plotOutDir, "avg_sizes_plot.png")
        print(f"saving plot to: {outPlot}")
        
        ids = [entry["id"] for entry in sizes]
        cm_areas = [entry["cmArea"] for entry in sizes]

        plt.figure(figsize=(10, 4))
        plt.plot(ids, cm_areas, marker="o", linewidth=2)
        plt.title("Average Pebble Size per Image")
        plt.xlabel("Image #")
        plt.ylabel("Average Size (cm²)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outPlot)
        plt.show()          # or plt.savefig("avg_sizes.png")
        return
    
    def chugSegment(self, image_file: str, imageID: int, outDir: str):
        """ 
        Used for testing a particular filter, although multiple filters can be tested 
        here modifying the values one at a time is best. 
        
        testVal is used when testing a particular filter it is in the format:
        [{"filter": <filerName>, "val": [value,value,value]}, etc...]
        
        modifyer values:
            minimumContours:[min_contours: int]
            minimumSize:    [min_area: int]
            touchingEdges:  [border_buffer: int]
            occluded:       [iou_thresh: float, overlap_self_thresh: float] 
            wholeness:      [min_solidity: float]
            convexHull:     [max_hull_diff: float, maxHullDiffRatio: float]
            complexity:     [epsilon_factor: float, min_vertices: int]
            roundish:       [min_roundness: float]       
            
        the float variables will need to be divided by 100 or a 1000 depending on use
        see the top of imgFilters.py for default values.   
        """
        
        imgFilters = ImageFilters()
        samProc = SAMprocess()    
            
        print(f"One centimeter = {self.oneCentimeter} pixels")
        print(f"processing: {image_file} to {outDir}")
        
        # 1. Initialization (Run SAM only once)
        mask_generator = samProc.load_sam()
        image, image_rgb = samProc.load_image(image_file)
        sam_masks = samProc.generate_masks(mask_generator, image_rgb)
        current_image = image
            
        # apply these filters, in this example we are testing roundish
        # "minimumSize" and "touchingEdges" work well and don't need adjustment
        filterList = ["minimumSize","touchingEdges", "roundish"]
            
        # set a range and step value as integers for the filename
        for pc in range (30, 100, 4):
                   
            # name the filter we are testing, with any value divison needed
            # testVar can be a list of filters. Those that are not listed use the
            # default values found at the top of imgFilters.py
            # the val can be a list of upto three modifyiers depending on the filter.
            tVal = [pc/100] # [pc/100, None, None]
            testVal = [{"filter": "roundish", "val": tVal }]
        
            filtered_masks, _ = imgFilters.applyfilters(image, sam_masks, filterList=filterList, testVal=testVal)
                    
            # Save the visualization
            output_image = samProc.makeOutputImage(current_image, filtered_masks)
            outFile = os.path.join(outDir, f"chug_{imageID:03d}_{pc:05d}.png")
            cv2.imwrite(outFile, output_image)
            padTval = imgFilters.getTestValues("test", [{"filter": "test", "val": tVal}])
            print (f"saved: {outFile}, values used: {padTval}")
        
        cv2.destroyAllWindows()        
        return
    
    def runSegment(self, image_file: str, interactive: bool=False):
        
        imgFilters = ImageFilters()
        imgUtilities = ImageUtilities()
        samProc = SAMprocess() 
        
        ## filters to be used
        # not used: "convexHull"
        filterList = ["minimumSize","touchingEdges","occluded", "wholeness", "complexity", "roundish"]
        
        print(f"processing: {image_file}")
        if interactive:
            try:
                samProc.run_interactive_segmentation(image_file)
            except Exception as e:
                print(f"An error occurred: {e}")
        else:        
            try:                        
                mask_generator = samProc.load_sam()
                image, image_rgb = samProc.load_image(image_file)
            
                sam_masks = samProc.generate_masks(mask_generator, image_rgb)

                # Process and filter the masks
                filtered_masks, pebble_data = imgFilters.applyfilters(image=image, sam_masks=sam_masks, filterList=filterList)

                # Calculate the final metrics   
                total_pebbles, average_size, average_wholeness = self.calculate_average_size_and_wholeness(pebble_data)

                # Display Results
                print(f"Image analyzed: {image_file}")
                print(f"Total fully-contained, non-overlapping pebbles counted: {total_pebbles}")
                print(f"Average size of measured pebbles: {average_size:.2f} pixels, {self.pxAreaToCM2(average_size):.2f} cm^2")
                print(f"Average wholeness score (Solidity): {average_wholeness:.3f} (closer to 1.0 is 'more whole')")

                output_image = samProc.makeOutputImage(image, filtered_masks)
                imgUtilities.showImage(output_image)
                # samProc.visualize_results(image, filtered_masks)
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                
        return
