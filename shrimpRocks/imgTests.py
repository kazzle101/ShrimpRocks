#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from shrimpRocks.imgUtilities import ImageUtilities
from shrimpRocks.getFiles import GetFiles

class ImageTests():

    def __init__(self, outDir):
        self.outDir = outDir
        self.imageTitle = "ImageTests"
        return
    
            
    def convexHullDifferenceTest(self, contour, convexHullDiff=None):
        """Creates an overlay showing deviations between the contour and its convex hull."""
        if convexHullDiff is None:
            convexHullDiff = self.CONVEX_HULL_DIFF
        
        hull_points = cv2.convexHull(contour, returnPoints=True)
        if len(hull_points) < 3:
            return None, {}

        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull_points)
        contour_perimeter = cv2.arcLength(contour, True)
        hull_perimeter = cv2.arcLength(hull_points, True)

        canvas_shape = (self.image_height, self.image_width)
        contour_mask = np.zeros(canvas_shape, dtype=np.uint8)
        hull_mask = np.zeros(canvas_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(hull_mask, [hull_points], -1, 255, thickness=cv2.FILLED)

        contour_region = contour_mask > 0
        hull_region = hull_mask > 0
        overlap = np.logical_and(contour_region, hull_region)
        hull_only = np.logical_and(hull_region, np.logical_not(contour_region))
        contour_only = np.logical_and(contour_region, np.logical_not(hull_region))

        overlay = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        overlay[overlap] = (0, 200, 0)       # shared area: green
        overlay[hull_only] = (0, 0, 200)     # hull beyond contour: blue
        overlay[contour_only] = (200, 0, 0)  # contour outside hull (should be empty)

        hull_defect_area = max(hull_area - contour_area, 0.0)
        hull_defect_ratio = hull_defect_area / (hull_area + 1e-6)
        perimeter_difference = contour_perimeter - hull_perimeter
        perimeter_defect_ratio = perimeter_difference / (hull_perimeter + 1e-6)

        print(f"perimeter_difference: {perimeter_difference:.3f}, perimeter_defect_ratio: {perimeter_defect_ratio:.3f}, {perimeter_difference > convexHullDiff}")

        metrics = {
            "contour_area": contour_area,
            "hull_area": hull_area,
            "hull_defect_area": hull_defect_area,
            "hull_defect_ratio": hull_defect_ratio,
            "contour_perimeter": contour_perimeter,
            "hull_perimeter": hull_perimeter,
            "perimeter_difference": perimeter_difference,
            "perimeter_defect_ratio": perimeter_defect_ratio,
        }
        
        return overlay, metrics
        
    
    
    def segment_pebbles(self, img_bgr, min_area_px=100):
        """
        Returns an int32 label image where 0 = background and 1..N are pebbles.
        More shadow-robust using LAB illumination normalization.
        """
        
        # --- 1) Flatten shadows/highlights on L channel (LAB) ---
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # smooth background illumination (big Gaussian) and normalize L by it
        bg = cv2.GaussianBlur(L, (0, 0), sigmaX=35, sigmaY=35)
        bg = np.clip(bg, 1, None)
        L_norm = cv2.divide(L, bg, scale=255)              # illumination corrected
        L_norm = cv2.bilateralFilter(L_norm, 7, 30, 7)     # denoise, keep edges
        L_eq   = cv2.createCLAHE(2.0, (8, 8)).apply(L_norm)

        # --- 2) Foreground mask (pebbles = white) ---
        _, th = cv2.threshold(L_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean()/255.0 < 0.35:                         # ensure pebbles = white
            th = cv2.bitwise_not(th)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, 1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k5, 2)

        # --- 3) Markers via distance transform (good seeds split touching pebbles) ---
        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        sure_fg = (dist > 0.40).astype(np.uint8) * 255      # ↑ raise if over-splitting
        sure_bg = cv2.dilate(clean, k3, 3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # --- 4) Watershed on a lightly blurred image ---
        markers_ws = cv2.watershed(cv2.GaussianBlur(img_bgr, (3, 3), 0), markers)
        labels = markers_ws.copy()
        labels[labels == -1] = 0

        # --- 5) Remove tiny flecks, fill tiny holes, and relabel 1..N ---
        relabeled = np.zeros_like(labels, dtype=np.int32)
        cur = 1
        for lbl in np.unique(labels):
            if lbl <= 1:
                continue
            comp = (labels == lbl).astype(np.uint8)
            if comp.sum() < min_area_px:
                continue
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k3, 1)
            relabeled[comp.astype(bool)] = cur
            cur += 1

        return relabeled
    
    def makeOutputImage(self, image, filtered_masks):
        """Draws the selected masks on the image and updates the specified window."""
        output_image = image.copy()
        
        outline_color = (0, 0, 255)  
        outline_thickness = 2        
        fill_color = (0, 255, 0) 
    
        overlay = np.zeros_like(output_image, dtype=np.uint8)
        
        # 1. Draw Fill
        for mask_data in filtered_masks:
            mask = mask_data['segmentation']
            overlay[mask] = fill_color

        alpha = 0.5 
        output_image = cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)

        # 2. Draw Outline
        for mask_data in filtered_masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, outline_color, outline_thickness)

        return output_image
    
    def makeTestOutput(self, image_file):  
        imageUtils = ImageUtilities()
              
        print(f"processing: {image_file} to {self.outDir}")
        
        try:
            img = cv2.imread(image_file)
        except Exception as e:
            print(f"CV2 Cannot load image: {image_file}")
            print(e)
            return None
        
        (height, width) = img.shape[:2]
        labels = self.segment_pebbles(img)
        imgOut = self.makeOutputImage(img, [{'segmentation': labels>0}])        
        imageUtils.showImage(imgOut, self.imageTitle, width, height+120)
    
        return