#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import os
import sys

# sudo apt install torch
#
# for nvidia cuda support:
# sudo apt remove torch
# install the cuda toolkit from nvidia first: https://developer.nvidia.com/cuda-downloads
# sudo pip install torch torchvision torchaudio

from shrimpRocks.imgFilters import ImageFilters
        
class SAMprocess:

    def __init__(self):
        self.checkpointPath = "sam_vit_h_4b8939.pth"  # use the checkpoint you have
        self.modelType = "vit_h"        
        self.windowName = 'Interactive Segmentation'
        self.llmPath = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
              
    def load_sam(self):
        # Initialize SAM
        self.checkpointCheck(self.checkpointPath)
        
        sam = sam_model_registry[self.modelType](checkpoint=self.checkpointPath)
        # Move SAM to the appropriate device (CPU/GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print (f"Using device: {device}")
        sam.to(device=device)

        # Initialize the mask generator
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator   

    def checkpointCheck(self, checkpointFile):        
        cwd = os.getcwd()        
        if not os.path.isfile(checkpointFile):
            print(f"Cannot load LLM: {checkpointFile}")
            print(f"Download it from: {self.llmPath} and copy to where you are running this program from ({cwd})")
            sys.exit()

    def load_image(self, image_path):
        """Loads the image and initializes the Segment Anything Model."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
       
        # Convert BGR to RGB (SAM expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb


    def generate_masks(self, mask_generator, image_rgb):
        """Generates all masks using SAM's automatic mask generator."""
        # The output is a list of dictionaries, each containing a segmentation mask
        sam_masks = mask_generator.generate(image_rgb)
        return sam_masks

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

