#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math 
import traceback
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

from shrimpRocks.imgUtilities import ImageUtilities
from shrimpRocks.imgFilters import ImageFilters
from shrimpRocks.samProcess import SAMprocess

class ClickImage:
    
    def __init__(self, oneCentimetre=75, outDir=None, font_size=24):
        self.oneCentimetre = oneCentimetre  # pixels 
        self.windowTitle = "Click Image"
        self.outDir = outDir
        self.font_size = font_size
        self._font_path = self._find_font_path()
        self._font_cache = {}
        return

    def _find_font_path(self) -> str | None:
        """
        Locate a TrueType font for crisp, readable text overlays.
        """
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            str(Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    _ = ImageFont.truetype(path, size=self.font_size)
                    return path
                except OSError:
                    continue
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        Return a cached PIL FreeType font, falling back to the default bitmap font.
        """
        key = int(size)
        if key not in self._font_cache:
            try:
                if self._font_path:
                    self._font_cache[key] = ImageFont.truetype(self._font_path, size=key)
                else:
                    raise OSError("No TrueType font available")
            except OSError:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]
    
    def drawAllOutlines(self, image, filtered_masks):
        
        output_image = image.copy()        
        outline_color = (0, 0, 255)  # BGR
        outline_thickness = 2        
               
        for mask_data in filtered_masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, outline_color, outline_thickness)

        return output_image

    def convexHullDiff(self, contour):
        
        hull_points = cv2.convexHull(contour, returnPoints=True)
        if len(hull_points) < 3:
            return 0,0,0,0
        
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull_points)
        contour_perimeter = cv2.arcLength(contour, True)
        hull_perimeter = cv2.arcLength(hull_points, True)

        hull_defect_area = max(hull_area - contour_area, 0.0)
        hull_defect_ratio = hull_defect_area / (hull_area + 1e-6)
        perimeter_difference = contour_perimeter - hull_perimeter
        perimeter_defect_ratio = perimeter_difference / (hull_perimeter + 1e-6)

        return {
            "contour_area": contour_area,
            "contour_perimeter": contour_perimeter,
            "hull_perimeter": hull_perimeter,
            "hull_area": hull_area,
            "hull_defect_area": hull_defect_area,
            "hull_defect_ratio": hull_defect_ratio,
            "perimeter_difference": perimeter_difference,
            "perimeter_defect_ratio": perimeter_defect_ratio 
        }
    
    def complexity(self, contour, epsilon_factor):
        
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)        
        num_vertices = len(approx)
        
        return {
            "perimeter": perimeter,
            "epsilon": epsilon,
            "num_vertices": num_vertices
        }

    def occluded(self, mask, exclusion_mask, iou_thresh, overlap_self_thresh):
        
        # work with boolean masks
        cur = (mask > 0)
        prev = (exclusion_mask > 0)

        inter = np.logical_and(cur, prev).sum()
        info = {
            "occluded": False,
            "iou": 0.0,
            "overlap_self": 0.0
        }
        if inter == 0:
            # no overlap: keep and update
            return info, cv2.bitwise_or(exclusion_mask, mask)

        area_cur = cur.sum()
        area_prev = prev.sum()
        union = area_cur + area_prev - inter

        iou = inter / (union + 1e-6)
        overlap_self = inter / (area_cur + 1e-6)
        is_occluded = (iou > iou_thresh) or (overlap_self > overlap_self_thresh)
        info.update({"occluded": is_occluded, "iou": iou, "overlap_self": overlap_self})
        if not is_occluded:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, mask)
        return info, exclusion_mask        
        

    def makeMaskEntries(self, image, filtered_masks, imgFilters):
        imgFilters = ImageFilters()
        
        exclusion_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        ## using the same default values as in ImageFilters
        iou_thresh = imgFilters.defaults['IOU_THRESH']
        overlap_self_thresh = imgFilters.defaults['OVERLAP_SELF_THRESH']
        epsilon_factor = imgFilters.defaults['EPSILON_FACTOR']
        
        mask_entries = []
        rng = np.random.default_rng(12345)
        for mask_data in filtered_masks:
            mask_bool = mask_data["segmentation"].astype(bool)
            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                        
            contours, _ = cv2.findContours(mask_uint8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = imgFilters.minimumContourFilter(contours)
            if contour is None:
                continue
            
            hull = self.convexHullDiff(contour)            
            solidity = hull["contour_area"] / (hull["hull_area"] + 1e-6)
            roundness = 0.0
            if hull["contour_perimeter"] > 0 and hull["contour_area"] > 0:
                roundness = 4 * math.pi * hull["contour_area"] / (hull["contour_perimeter"] ** 2)
            bbox = cv2.boundingRect(contour)
            color = tuple(int(c) for c in rng.integers(90, 255, size=3))

            occlusion_info, exclusion_mask = self.occluded(mask_uint8, exclusion_mask, iou_thresh, overlap_self_thresh)
            complex = self.complexity(contour, epsilon_factor)
            
            mask_entries.append(
                {
                    "mask": mask_bool,
                    "bbox": bbox,
                    "color": color,                    
                    
                    "contour": contour,
                    "contour_points": len(contour),
                    "contour_area": hull["contour_area"],
                    "contour_perimeter": hull["contour_perimeter"],
                    "perimeter_diff": hull["perimeter_difference"],
                    "solidity": solidity,
                    "hull_diff": hull["hull_defect_area"],
                    "hull_diff_ratio": hull["hull_defect_ratio"],
                    "roundness": roundness,
                    
                    "perimeter": complex["perimeter"],
                    "epsilon": complex["epsilon"],
                    "num_vertices": complex["num_vertices"],
                    
                    "occluded": occlusion_info["occluded"],
                    "iou": occlusion_info["iou"],
                    "overlap_self": occlusion_info["overlap_self"]
                }
            )
        
        return mask_entries
        
    def draw_text_block(self, img, lines, origin=(10, 30)):
        # Draw text using PIL with TrueType fonts for better readability.
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        x, y = origin
        font = self._get_font(self.font_size)
        font_px = getattr(font, "size", self.font_size)
        line_spacing = max(int(font_px * 1.4), int(self.font_size * 1.6))
        shadow_offset = 1
        for line in lines:
            shadow_pos = (x + shadow_offset, y + shadow_offset)
            draw.text(shadow_pos, line, font=font, fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=(240, 240, 240))
            y += line_spacing
            
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def info_lines_extend(self, info_lines, highlighted_idx, entry, defaults):
        
        info_lines.extend(
            [
                f"Mask #{highlighted_idx + 1}",
                f"Contour points: {entry['contour_points']} (>= {defaults['MIN_CONTOURS']})",
                " ",
                f"Area: {entry['contour_area']:.1f} px^2 (>= {defaults['MIN_AREA']})",
                f"Solidity: {entry['solidity']:.3f} (>= {defaults['MIN_SOLIDITY']:.2f})",
                f"Perimeter diff: {entry['perimeter_diff']:.1f} (<= {defaults['CONVEX_HULL_DIFF']})",
                f"Hull diff ratio: {entry['hull_diff_ratio']:.3f} (<= {defaults['MAX_HULL_DIFF_RATIO']:.3f})",
                f"Hull area diff: {entry['hull_diff']:.1f} px^2",
                f"Roundness: {entry['roundness']:.3f} (>= {defaults['MIN_ROUNDNESS']:.2f})",
                "Occluded: ",
                f" iou: {entry['iou']:.2f}  (>{defaults['IOU_THRESH']:.2f})",
                f" overlap_self: {entry['overlap_self']:.2f} (>{defaults['OVERLAP_SELF_THRESH']:.2f})",
                "Complexity: ",
                f" perimeter: {entry['perimeter']:.2f}",
                f" epsilon: {entry['epsilon']:.2f}",
                f" num_vertices: {entry['num_vertices']} (>= {defaults['MIN_VERTICES']})",
                ]
            )
        
        return info_lines
   
    def makeClickImage(self, image_file):
        
        imgFilters = ImageFilters()
        imgUtilities = ImageUtilities()
        samProc = SAMprocess()   
        
        screen_width, screen_height = imgUtilities.getCurrentScreenRes()
        ## filters to be used
        filterList = ["touchingEdges", "minimumSize", "occluded", "wholeness",]#, "convexHull", "complexity", "roundish"]
        
        print(f"loading image: {image_file}")
        # 1. Initialization (Run SAM only once)
        mask_generator = samProc.load_sam()
        image, image_rgb = samProc.load_image(image_file)
        sam_masks = samProc.generate_masks(mask_generator, image_rgb)
        current_image = image.copy()
        
        filtered_masks, _ = imgFilters.applyfilters(image, sam_masks, filterList=filterList)
        current_image = self.drawAllOutlines(image, filtered_masks)
        mask_entries = self.makeMaskEntries(image, filtered_masks, imgFilters)
        if not mask_entries:
            print("No valid masks available for interaction; showing the original image.")
            imgUtilities.showImage(current_image, self.windowTitle)
            return

        # get the default values from ImageFilters, used in the output.
        filter_defaults = imgFilters.defaults

        window_name = self.windowTitle
        instructions = ["Left click a pebble to inspect", "Press 'q' or 'Esc' to close"]
        highlighted_idx = None
        display_image = current_image.copy()
        panel_width = 460
        screen_width, screen_height = imgUtilities.getCurrentScreenRes()
        window_dims: tuple[int, int] | None = None

        def render_display():
            nonlocal window_dims
            vis = current_image.copy()
            info_lines = list(instructions)
            panel = np.full((vis.shape[0], panel_width, 3), (32, 32, 32), dtype=np.uint8)

            if highlighted_idx is not None and 0 <= highlighted_idx < len(mask_entries):
                entry = mask_entries[highlighted_idx]
                overlay = np.zeros_like(vis)
                overlay[entry["mask"]] = entry["color"]
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0.0)
                cv2.drawContours(vis, [entry["contour"]], -1, (0, 255, 0), 2)
                x, y, w, h = entry["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
                info_lines = self.info_lines_extend(info_lines, highlighted_idx, entry, filter_defaults)

            panel = self.draw_text_block(panel, info_lines)
            combined = np.concatenate((vis, panel), axis=1)
            combined_h, combined_w = combined.shape[:2]
            
            # Scale the window slightly larger than the content without oversizing
            scale_w = screen_width * 0.9 / combined_w
            scale_h = screen_height * 0.9 / combined_h
            scale_factor = max(min(1.08, scale_w, scale_h), 0.1)
            target_w = int(combined_w * scale_factor)
            target_h = int(combined_h * scale_factor)
            window_dims = (target_w, target_h)
            return combined

        def on_mouse(event, x, y, _flags, _param):
            nonlocal highlighted_idx, display_image
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            selected = None
            for idx in range(len(mask_entries) - 1, -1, -1):
                mask = mask_entries[idx]["mask"]
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    selected = idx
                    break

            highlighted_idx = selected
            display_image = render_display()
            if window_dims:
                win_w, win_h = window_dims
                cv2.resizeWindow(window_name, win_w, win_h)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 50)
        cv2.setMouseCallback(window_name, on_mouse)

        display_image = render_display()
        if window_dims:
            win_w, win_h = window_dims
            cv2.resizeWindow(window_name, win_w, win_h)
        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break

        cv2.destroyWindow(window_name)
        return
