# ShripRocks

In September 2024 Mike on his Atomic Shrimp YouTube channel went to Chesil Beach in Dorset on the south coast of England to measure the pebbles and show tht they get larger as he traversed the 18 mile long tombolo from Burton Bradstock in the North West to Fortuneswell in the South East by using the method of measuring a sample every 500 meters and writing down the results in a notebook with a pencil. In the video he challenged programmers to do this in code, here is my attempt.

The Atomic Strimp Video, Chesil Beach Pebble Survey Part 1 - Spending the Whole Day Looking at Gravel: https://www.youtube.com/watch?v=8RAuvyWM_2E

Each image is treated individually, no assumptions or expectations are applied as to the size of the pebbles in each image. 

## Options
```
$ python shrimpRocks.py --help
usage: shrimpRocks.py [-h] [-p] [-a] [--croptest CROPTEST] [--segment SEGMENT] [--chug CHUG] [--makereadme MAKEREADME] [--clickimage CLICKIMAGE]

Futility for measuring pebble sizes on Chesil Beach. Image numbers are in the range 1 to 33 and correspond to those found in the images/source/ or images/cropped/ directories

options:
  -h, --help            show this help message and exit
  -p, --process         Crop the original images ready for examination
  -a, --averagesize     Show the average pebble sizes and output a plot
  --croptest CROPTEST   Use the image number to test an individual image, for checking the crop process is working
  --segment SEGMENT     Filer Test, using the image number display an indivdual rock image with the filters applied
  --chug CHUG           Filter Test, use an image number for testing a filter with a range of values, files are output to
                        images/test/
  --makereadme MAKEREADME
                        Make images for the readme.md file using an image number
  --clickimage CLICKIMAGE
                        Using an image number, loads a filtered image, allows you to click on the masks for information
                        about the mask
```

## Image Processing
### Images are first processed with:
```
python shrimpRocks.py --p
```
We start with a selection of screen grabs, these can be found in the in the images/source directory

From this OpenCV is used to crop away the GPS tracking data and find the inside top left of the ruler by detecting the horizontals and verticals, here shown as green and red lines:


And from there the image is cropped a fixed width and height and saved to the images/cropped directory

### Find the average pebble size 
```
python shrimpRocks.py --a
```
We now need to identify each pebble, much time was spent trying to have OpenCV do this, see imgTests.py but it gets confused by the shadows and as these change as Mike perambulated along the beach during the day the settings would need adjustment for each image. After much experimentation I settled on using the Segment Anything AI from Meta. 

This creates a mask with all the pebbles selected:


and from that number of different filters (in imgFilters.py) are applied to remove pebbles from the sample that do not qualify, some with more sucesss than others, the settings for each filter is a compromise so to have them work across the range of pebble sizes.

Those that are overlap the image edge are removed.

Those where the outline (contour) is too short, or that the area they take up is too small

Attempt to remove those that are overlapping/occluded 

Attempt to remove those with too high a convex edge, as these will probably be overlapping

Attempt to remove those with a complex shape, as again these will probably be underneath other pebbles.


each pebbles area is given in pixles, this is converted to cm^2 with some maths, looking on the original images with the ruler I measured one centemeter to approimate 56 pixels in length.


With all that, I can confirm that the pebbles do, indeed, get larger as you traverse the beach from North West to South East. 


On the images with the larger rocks, I think the sample size is confusing the results

## Installation and Dependancies
I wrote this Python using a Debian Linux (trixie) terminal running in a WSL on my Windows computer, Nvidia CUDA gives a speed boost but not a requirement. As is traditional with these kind of instructions, some elements may be alreay installed or missing. This is my setup, it should work for any deb based distro, and only minor changes should be needed for the likes of Redhat syle distros.

The apt package for OpenCV does not make use of CUDA, and installing it manually would be a lot of words.
```
sudo apt install python3-opencv python3-natsort python3-scipy python3-sklearn python3-skimage
sudo apt install torch python3-onnxruntime python3-onnx python3-torchvision
sudo pip install opencv-contrib-python --break-system-packages
```
Install Segment Anything:
```
cd ~/Downloads
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e . --break-system-packages
```
copy this file to where you installed ShrimpRocks, it is the LLM for Segment Anything:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
for nvidia cuda support in torch the debian package does not include support:
install the cuda toolkit from nvidia first: https://developer.nvidia.com/cuda-downloads
```
sudo apt remove torch
sudo pip install torch torchvision torchaudio --break-system-packages
```
