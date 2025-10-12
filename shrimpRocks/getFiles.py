#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from natsort import natsorted
import json

class GetFiles():
    
    def __init__(self):
        return
    
    def filesList(self, imageDir: str) -> list:
        
        images = []
        if not os.path.isdir(imageDir):
            print(f"Image diretory not found: {imageDir}")
            return None 
    
        for file in os.listdir(imageDir):
            if file.endswith(".png"):
                images.append(os.path.join(imageDir, file))
       
        if len(images) < 1:
            print(f"PNG Images not found (empty directory)")
            return None
            
        return natsorted(images)
    
    def makeOutputDir(self, outputDir: str):
        
        try:
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)
        except Exception as e:
            print(f"Cannot create directory: {outputDir}")
            print(e)
            sys.exit()
            
        return
    
    def deleteFiles(self, delDir: str):
        
        files = self.filesList(delDir)
        if files is None:
            return
        
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Cannot delete file: {file}")
                print(e)
                sys.exit()
        
        return
    
    def isRockfordFile(self, files: list, id: int) -> str:
        
        if files is None:
            return None
        
        try:
            filename = files[id-1]
        except:
            return None
            
        return filename
    
    def saveSettings(self, filename: str, data: list):
        
        if data is None or not data:
            return
        
        jData = json.dumps(data, indent=4)
        
        try:
            with open(filename, "w") as textFile:
                textFile.write(jData)
        except Exception as e:
            print(f"Cannot write to file: {filename}")
            print(e)
            sys.exit()  
        
        print (f"settings file saved: {filename}")            
        return
    
    def getSettings(self, filename: str) -> list:
        
        if not os.path.isfile(filename):
            print(f"settings file not found: {filename}")
            return None

        try:         
            with open(filename) as file:
                data = json.loads(file.read())
        except Exception as e:
            print(f"Cannot load settings file: {filename}")
            print(e)
            return None
            
        return data