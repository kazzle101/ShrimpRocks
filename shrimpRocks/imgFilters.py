import cv2
import numpy as np


class ImageFilters():
    """ 
    a selection of filters to remove pebble masks that do not qualify.
    """
    
    def __init__(self):
        
        ## these are the default values
        self.defaults = {
            "MIN_CONTOURS": 85,             # contourCheck
            "MIN_AREA": 3000,               # minimumSizeFiler,  Minimum area for a contour to be considered valid
            "BORDER_BUFFER": 5,             # touchingEdges
            "IOU_THRESH": 0.5,              # occlusionMask
            "OVERLAP_SELF_THRESH": 0.15,    # occlusionMask
            "MIN_SOLIDITY": 0.15,           # wholenessScore
            "MAX_DEFECT_RATIO": 54 / 1000,  # convexShapeFilter, 0.054% threshold for the defect ratio
            "EPSILON_FACTOR": 0.02,         # complexShapeFilter, 
            "MIN_VERTICES": 7,              # complexShapeFilter, Minimum number of vertices for complex shape filter
            "CONVEX_HULL_DIFF": 24,         # convexHullDifference, Maximum allowed difference between contour and convex hull areas
            "MAX_HULL_DIFF_RATIO": 0.030,   # convexHullDifference
            "MIN_ROUNDNESS": 0.35           # is_roundish
        }    
     
    def minimumContourFilter(self, contours, min_contours: int=None) -> list[np.int32]:
        """
        Checks if the contour is valid (at least self.MIN_CONTOURS points).
        Selects the largest contour from those contours that make a mask
        """
        if min_contours is None:
            min_contours = self.defaults['MIN_CONTOURS']
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)   # best single-contour choice
        if len(contour) < min_contours:
            return None
        
        return contour
    
    def minimumSizeFilter(self, contour, area, min_area: int=None) -> bool: #, min_contours=None):
        """
        Filters contours based on area and perimeter.
        """
        if min_area is None:
            min_area = self.defaults['MIN_AREA']
            
        # if min_contours is None:
        #     min_contours = self.MIN_CONTOURS
        
        # if len(contour) < min_contours:
        #     return False    
            
        perimeter = cv2.arcLength(contour, True)
        # print(f"min_area: {min_area},  Area: {area}, Perimeter: {perimeter}, {(self.MIN_AREA < area and perimeter > 0)}")   
        return (min_area < area and perimeter > 0)        
    
    def touchingEdges(self, mask, height: int, width: int,  border_buffer: int=None) -> bool:
        """
        Checks if the mask touches the edges of the image within a specified buffer.
        """
        if border_buffer is None:
            border_buffer = self.defaults['BORDER_BUFFER']
        
        x, y, w, h = cv2.boundingRect(mask)
        touches_edge = (x < border_buffer or y < border_buffer or 
                        (x + w) > (width - border_buffer) or 
                        (y + h) > (height - border_buffer))
        return touches_edge
    
    def occlusionMask(self, mask, exclusion_mask, iou_thresh: float=None, overlap_self_thresh: float=None) -> tuple:
        """ 
        decides if a candidate pebble mask overlaps too much with ones already accepted
        """
        if iou_thresh is None:
            iou_thresh = self.defaults['IOU_THRESH']
        if overlap_self_thresh is None:
            overlap_self_thresh = self.defaults['OVERLAP_SELF_THRESH']
        
        # work with boolean masks
        cur = (mask > 0)
        prev = (exclusion_mask > 0)

        inter = np.logical_and(cur, prev).sum()
        if inter == 0:
            # no overlap: keep and update
            return False, cv2.bitwise_or(exclusion_mask, mask)

        area_cur = cur.sum()
        area_prev = prev.sum()
        union = area_cur + area_prev - inter

        iou = inter / (union + 1e-6)
        overlap_self = inter / (area_cur + 1e-6)

        occluded = (iou > iou_thresh) or (overlap_self > overlap_self_thresh)
        if not occluded:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, mask)
        return occluded, exclusion_mask        
                
    def wholenessScore(self, contour, area, minSolidity: float=None) -> tuple:
        """ 
        Wholeness Score (Solidity) Check 
        """
        if minSolidity is None:
            minSolidity = self.defaults['MIN_SOLIDITY']    
                    
        # Get indices for defects (hull) and points for area calculation (hull_points)
        # hull = cv2.convexHull(contour, returnPoints=False)
        
        # CRITICAL CHECK: Must have at least 3 points for a valid hull
        hull_points = cv2.convexHull(contour, returnPoints=True)        
        if len(hull_points) < 3:
            return False, 0
        
        # Solidity Calculation (Wholeness Score)
        # FIX: Use hull_points (the array of points) for contourArea, not hull (the array of indices)
        hull_area = cv2.contourArea(hull_points) 
        if hull_area == 0: 
            return False, 0
            
        solidity = area / hull_area # This is our Wholeness Score            
        if solidity < minSolidity: 
            return False, solidity
            
        return True, solidity
        
    def convexShapeFilter(self, contour, maxDefectRatio: float=None) -> bool:
        """
        Filters a contour if its non-convex perimeter deviation exceeds the max defect ratio.
        """
        if maxDefectRatio is None:
            maxDefectRatio = self.defaults['MAX_DEFECT_RATIO']
        
        # 1. Calculate perimeters
        hull_points = cv2.convexHull(contour, returnPoints=True)
        
        # Guard against small contours/hulls (must have at least 3 points for a valid hull)
        if len(hull_points) < 3:
            return False 

        contour_perimeter = cv2.arcLength(contour, True)
        hull_perimeter = cv2.arcLength(hull_points, True)
            
        # The total deviation from convexity is the difference in perimeters
        deviation_length = contour_perimeter - hull_perimeter
        # Check the Ratio, hull_perimeter is guaranteed > 0 by the len(hull_points) check
        defect_ratio = deviation_length / hull_perimeter

        # Filter if the total non-convex perimeter is too high
        # print(f"Defect Ratio: {defect_ratio:.3f} (Max allowed: {maxDefectRatio:.3f})")
        if defect_ratio > maxDefectRatio:
            return False

        return True        

    def convexHullDifference(self, contour, convexHullDiff: int=None, maxHullDiffRatio: float=None)-> bool:
        """
        The greater the diference between the contour and its convex hull indicates a 
        non-convex (pebble covered by another) shape.
        """
        if convexHullDiff is None:
            convexHullDiff = self.defaults['CONVEX_HULL_DIFF']
            
        if maxHullDiffRatio is None:
            maxHullDiffRatio = self.defaults['MAX_HULL_DIFF_RATIO']
                    
        hull_points = cv2.convexHull(contour, returnPoints=True)
        if len(hull_points) < 3:
            return False

        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull_points)
        contour_perimeter = cv2.arcLength(contour, True)
        hull_perimeter = cv2.arcLength(hull_points, True)

        hull_defect_area = max(hull_area - contour_area, 0.0)
        hull_defect_ratio = hull_defect_area / (hull_area + 1e-6)
        perimeter_difference = contour_perimeter - hull_perimeter
        perimeter_defect_ratio = perimeter_difference / (hull_perimeter + 1e-6)
        
        # print(f"{hull_defect_ratio} > {maxHullDiffRatio} = {hull_defect_ratio > maxHullDiffRatio}")
        if hull_defect_ratio > maxHullDiffRatio:
            return False
        
        # if perimeter_difference > convexHullDiff:
        #     return False

        return True

    def complexShapeFilter(self, contour, epsilon_factor: float=None, min_vertices: int=None) -> bool:
        """
        Filters a contour if its complexity (number of vertices after approximation) exceeds 
        the maximum allowed.
        """
        if epsilon_factor is None:
            epsilon_factor = self.defaults['EPSILON_FACTOR']
                        
        if min_vertices is None:
            min_vertices = self.defaults['MIN_VERTICES']  
                        
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Determine epsilon based on the perimeter
        epsilon = epsilon_factor * perimeter
        
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check the number of vertices in the approximated contour
        num_vertices = len(approx)
        
        # Filter if the number of vertices is less than the minimum allowed
        # print(f"Num Vertices: {num_vertices} (Min allowed: {min_vertices}), {(num_vertices > min_vertices)}")
        return (num_vertices > min_vertices)

        
    def is_roundish(self, mask_bool, min_roundness: float=None) -> bool:
        """
        See how round the selected pebble is, to filter out irregular shapes
        """        
        # Roundness = 4πA / P² -> ~1 for perfect circle, smaller for irregular
        if min_roundness is None:
            min_roundness = self.defaults['MIN_ROUNDNESS']
        
        m = (mask_bool.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            return False
        
        c = max(cnts, key=cv2.contourArea)
        A = cv2.contourArea(c)
        P = cv2.arcLength(c, True)
        if P <= 0 or A <= 0: 
            return False
        
        roundness = 4 * np.pi * A / (P * P)
        return roundness > min_roundness  # tune threshold based on your stones
    
    def getTestValues(self, filter: str, testVal: list) -> list:
        """
        Used when testing a filter, it checks for any values in a given filter and returns 
        a list padded to three items, see chugSegment in imgAnalyse.py 
        """
        pad = 3
        outVal = []
        if not testVal:
            return (outVal + pad * [None])[:pad]

        if filter == "test":
            outVal = testVal[0]["val"]
            return (outVal + pad * [None])[:pad]

        for t in testVal:
            if t["filter"] == filter:
                outVal = t["val"]
                break
                
        ## return a list padded to three items
        return (outVal + pad * [None])[:pad]
    
    
    # available filers:
    # filterList = ["minimumSize","touchingEdges","occluded", "wholeness",
    #               "convexHull", "conplexity", "roundish"]
    # the optional testVal
    def applyfilters(self, image: list[np.ndarray], sam_masks: list, filterList: list, testVal: list = []) -> tuple:
        """
        Apply the filters, available filters:
        ["minimumSize", "touchingEdges", "occluded", "wholeness", "convexHull", "complexity", "roundish"]
        
        Optional: testVal is used when testing a particular filter it is in the format:
        [{"filter": <filerName>, "val": [value,value,value]}, etc...]
        
        modifyer values:
            minimumContours: [min_contours: int]
            minimumSize:     [min_area: int]
            touchingEdges:   [border_buffer: int]
            occluded:        [iou_thresh: float, overlap_self_thresh: float]
            wholeness:       [min_solidity: float]
            convexHull:      [max_hull_diff: float, maxHullDiffRatio: float]
            complexity:      [epsilon_factor: float, min_vertices: int]
            roundish:        [min_roundness: float]            
        """
    
        filtered_masks = []
        pebble_data = [] 
        
        height, width = image.shape[:2]
        exclusion_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Sort masks by area (smallest first)
        sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=False)
        
        for mask_data in sorted_masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']
        
            # Contour Extraction
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
               continue
      
            min_contours, _, _ = self.getTestValues("minimumContours", testVal)
            contour = self.minimumContourFilter(contours, min_contours)
            if contour is None:
                continue
            
            if "minimumSize" in filterList:
                min_area, _, _ = self.getTestValues("minimumSize", testVal)
                if not self.minimumSizeFilter(contour, area, min_area):
                    continue
        
            # Edge Proximity Check
            if "touchingEdges" in filterList: 
                border_buffer, _, _ = self.getTestValues("touchingEdges", testVal)           
                if self.touchingEdges(mask, height, width, border_buffer):
                    continue
                    
            if "occluded" in filterList:
                iou_thresh, overlap_self_thresh, _ = self.getTestValues("occluded", testVal)
                occluded, exclusion_mask = self.occlusionMask(mask, exclusion_mask, iou_thresh, overlap_self_thresh)
                if occluded:
                    continue
                    
            _ , solidity = self.wholenessScore(contour, area)
            # Wholeness Score (Solidity) Check
            if "wholeness" in filterList:
                minSolidity, _, _ = self.getTestValues("wholeness", testVal)
                wholeness, _ = self.wholenessScore(contour, area, minSolidity)
                if not wholeness:
                    continue     
              
            # this filters out too many large pebbles  
            if "convexHull" in filterList:
                convexHullDiff, maxHullDiffRatio, _ = self.getTestValues("convexHull", testVal)
                if not self.convexHullDifference(contour, convexHullDiff, maxHullDiffRatio):
                    continue
                
            if "complexity" in filterList:
                epsilon_factor, min_vertices, _ = self.getTestValues("complexity", testVal)
                if not self.complexShapeFilter(contour, epsilon_factor, min_vertices):
                    continue
                
            if "roundish" in filterList:
                min_roundness, _, _ = self.getTestValues("roundish", testVal)
                if not self.is_roundish(mask, min_roundness):
                    continue
                
            filtered_masks.append(mask_data)
            pebble_data.append((area, solidity))
        
        return filtered_masks, pebble_data
    

        
        
