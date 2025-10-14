#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import tkinter as tk

class ImageUtilities:
    
    def __init__(self):
        return

    def showImage(self, img: list[np.ndarray], imgTitle: str="CV2 Image", width: int=None, height: int=None):
        
        if img is None:
            print ("no image to display")
            return
    
        if width is None or height is None:
            height, width = img.shape[:2]
        
        print(f"To exit, press any key while in the {imgTitle} window")
        #img = cv2.resize(img, (width, height))      
        
        cv2.namedWindow(imgTitle, cv2.WINDOW_NORMAL)        
        cv2.moveWindow(imgTitle, 100, 50)
        cv2.resizeWindow(imgTitle, (width, height))        
        cv2.imshow(imgTitle, img)
        cv2.waitKey(2)
        while True:
            if cv2.getWindowProperty(imgTitle, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            key = cv2.waitKey(1)
            if key != -1: 
                break
            
        cv2.destroyAllWindows()       
        return
    
    def saveImage(self, filename: str, image: list[np.ndarray]):
        
        # print(filename)
        
        if os.path.exists(filename):
            os.remove(filename)
        
        try:            
            cv2.imwrite(filename, image)
        except Exception as e:
            print(f"Cannot save image: {filename}")
            print(e)
            
        return
        
    def setWindowScaleImage(self, image: list[np.ndarray]) -> tuple:
        
        # set the scaling for 4K monitors, half that for everything else
        (w, h) = self.getCurrentScreenRes()
        # print(f"screen Width:{w}, Height:{h}")        
        if w > 3000:
            scale = .9
        else:
            scale = .5
        
        (h, w) = image.shape[:2]

        width = int((w*2) * scale)
        height = int((h) * scale) 
        return width, height, image
        
    def setWindowScale(self, imagePath: str) -> tuple:
        
        try:
            img = cv2.imread(imagePath)
        except Exception as e:
            print(f"CV2 Cannot load image: {imagePath}")
            print(e)
            return None, None, None
    
        return self.setWindowScaleImage(img)
        
        
    def concat_same_height(self, left_img, right_img):
        if left_img is None or right_img is None:
            raise ValueError("Both images must be provided.")

        def normalize(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

        left = normalize(left_img)
        right = normalize(right_img)

        target_height = min(left.shape[0], right.shape[0])
        if target_height <= 0:
            raise ValueError("Images must have positive height.")

        def resize_to_height(img, height):
            if img.shape[0] == height:
                return img
            scale = height / img.shape[0]
            new_width = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_width, height), interpolation=cv2.INTER_LINEAR)

        left_resized = resize_to_height(left, target_height)
        right_resized = resize_to_height(right, target_height)

        gap_width = 20
        gap = np.zeros((target_height, gap_width, left_resized.shape[2]), dtype=left_resized.dtype)

        return np.concatenate((left_resized, gap, right_resized), axis=1)
        
        
    def show(self, imgs: list[np.ndarray], imgTitle: str, width: int, height: int):
            
        imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in imgs]
        if len(imgs) == 1:
            imgConcat = imgs[0]
            imgConcat = cv2.resize(imgConcat, (width, height), interpolation= cv2.INTER_LINEAR)        
        elif len(imgs) == 3:
            (h, w) = imgs[2].shape[:2]            
            firstTwo = np.concatenate(imgs[:2], axis=1)  # Concatenate the first two images horizontally
            imgConcat = np.concatenate([firstTwo, imgs[2]], axis=0)  # Concatenate the third image below
            imgConcat = cv2.resize(imgConcat, (width, height+h), interpolation= cv2.INTER_LINEAR)
        else:
            imgConcat = np.concatenate(imgs, axis=1)
            imgConcat = cv2.resize(imgConcat, (width, height), interpolation= cv2.INTER_LINEAR)

        cv2.waitKey(1)
        cv2.imshow(imgTitle, imgConcat)
        return

    def getCurrentScreenRes(self) -> tuple:
        root = tk.Tk()
        root.withdraw()  # Hides the tkinter window
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        return width, height    

    def pxAreaToCM2(self, pxArea: float, oneCentimetre: int) -> float:
        """Converts pixel area to square centimeters."""
        cmArea = (pxArea / (oneCentimetre * oneCentimetre))        
        return cmArea        

