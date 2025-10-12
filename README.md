# ShrimpRocks
In September 2024, Mike on his Atomic Shrimp YouTube channel went to Chesil Beach in Dorset on the south coast of England, to demonstrate that the pebbles get larger along the length of the 18 mile long tombolo from Burton Bradstock (North West) to the Isle of Portland (South East). He challenged programmers to automate this analsys> This project is my attempt, using still images taken from the video to measure the average surface area of the pebbles in cm<sup>2</sup>, rather than phyiscally evaluating a sample.

_The Atomic Strimp Video, Chesil Beach Pebble Survey Part 1 - Spending the Whole Day Looking at Gravel:_ <a href='https://www.youtube.com/watch?v=8RAuvyWM_2E' target='_blank'>https://www.youtube.com/watch?v=8RAuvyWM_2E</a>

Each image is treated individually, no assumptions or expectations are applied as to the size of the pebbles in each image. 

## Installation and Dependancies
I wrote this in Python using a Debian Linux (trixie) terminal running in a WSL on my Windows computer, Nvidia CUDA gives a speed boost is but not a requirement. As is traditional with these kind of instructions, some elements may be alreay installed or missing. This is my setup, it should work for any deb based distro and only minor changes should be needed for the likes of Redhat syle distros. The apt package for OpenCV does not make use of CUDA, and installing it manually would be out of scope here.
```
sudo apt install python3-opencv python3-natsort python3-scipy python3-sklearn python3-skimage
sudo apt install torch python3-onnxruntime python3-onnx python3-torchvision
sudo pip install opencv-contrib-python --break-system-packages
```
#### Install the Segment Anything AI:
```
cd ~/Downloads
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e . --break-system-packages
```
You will also need the LLM, download and copy this file to the root of where you installed ShrimpRocks:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Optional, and you have a suitable GPU: As the apt package for torch does not include CUDA support, this is easy to enable and gives Segment Anything a speed boost:
for your platform, install the cuda toolkit from nvidia first: <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">https://developer.nvidia.com/cuda-downloads</a>
```
sudo apt remove torch
sudo pip install torch torchvision torchaudio --break-system-packages
```
#### Install ShrimpRocks
```
cd ~
git clone https://github.com/kazzle101/ShrimpRocks.git
cd ShrimpRocks
python shrimpRocks.py --help
```

### Usage and Options
```
$ python shrimpRocks.py --help
usage: shrimpRocks.py [-h] [-p] [-a] [--croptest CROPTEST] [--segment SEGMENT] [--chug CHUG] [--makereadme MAKEREADME] [--clickimage CLICKIMAGE]

Futility for measuring pebble sizes on Chesil Beach. Image numbers are in the range 1 to 33 and correspond to those found in the images/source/ or images/cropped/ directories

options:
  -h, --help            show this help message and exit
  -p, --process         Crop the original images ready for examination.
  -a, --averagesize     Show the average pebble sizes and output a plot after the images have been processed.
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
Note: existing images created with the options: `--process, --averagesize, --segment, --chug and --makereadme` are automatically deleted before the new files are written.

## Image Processing and Measuring
#### 1. Images are first processed with:
```
python shrimpRocks.py --p
```
__Source Images:__ We start with a selection of screen grabs, these can be found in the in the `images/source` directory:

<img src='./images/readmeImgs/01_source_image.png?raw=true' alt="Source Image" width='800' />

__Ruler Dectection:__ OpenCV is used to crop away the GPS tracking data and find the inside top left of the ruler by detecting the horizontals and verticals, here shown as green and red lines:

<img src='./images/readmeImgs/02_rulers_selected.png?raw=true' alt="Rulers Selected" width='400' />

__Cropping:__ The image is cropped a fixed width and height and saved to the `images/cropped` directory

<img src='./images/readmeImgs/03_source_pebbles.png?raw=true' alt="Source Pebbles" width='300' />

#### 2. Find the average pebble size:
```
python shrimpRocks.py --a
```
__Initial Segmentation (Segment Anything AI):__ In this stage the quest is to select a sutiable range of pebbles in each image, preferably those that are on the surface. Once selected we can obtain the average area of the selected pebbles in pixels<sup>2</sup> and convert that to Centimetre<sup>2</sup>, much time was spent trying to have OpenCV do this nativley (see imgTests.py) but it gets confused by the shadows and struggled to select a whole pebble. After much experimentation I settled on using the Segment Anything AI from Meta. Using this we create a mask with all the pebbles selected:

<img src='./images/readmeImgs/04_all_pebbles_selected.png?raw=true' alt="All Pebbles Selected" width='300' />

__Applying Filters:__ A number of different filters (in `imgFilters.py`) are applied to remove pebbles from the sample that do not qualify, some with more sucesss than others, the settings for each filter is a compromise so to have them work across the range of pebble sizes. First, those where the outline (contour) is too short, or that the area they take up is too small are removed:

<img src='./images/readmeImgs/05_filter_minimum_size.png?raw=true' alt="Remove pebbles that are too small" width='300' />

__Edge Overlap Filter:__ Those that are overlaping the image edge are removed:

<img src='./images/readmeImgs/06_filter_touching_edge.png?raw=true' alt="Remove those that overlap the edge" width='300' />

__Overlap Filter:__ Removes those that are overlapping/occluded, the suns shadow or a stain on a pebble can result in a mask within a mask, these are removed:

<img src='./images/readmeImgs/07_filter_occluded.png?raw=true' alt="Remove those that overlap the edge" width='300' />

__Wholeness Filter:__ Check for each mask for wholeness, see how solid it is, this example is a bit poor, it works bettwer with different pebble sizes:

<img src='./images/readmeImgs/08_filter_wholeness.png?raw=true' alt="Wholeness Filter" width='300' />

__Convex Hull Filter:__ Attempt to remove those with too high a convex edge, as these will probably be overlapping, this is the amount of difference beween the masked object and the outline of a rubber band streched around it:

<img src='./images/readmeImgs/09_filter_convex_hull.png?raw=true' alt="Convex Hull Filter" width='300' />

__Complexity Filter__ Attempts to remove those with a complex shape, as again these will probably be underneath other pebbles:

<img src='./images/readmeImgs/10_filter_complexity.png?raw=true' alt="Complexity Filter" width='300' />

__Roundness Filter:__ Removes objects that are mostly square, as these are also likely to be underneath other pebbles.

<img src='./images/readmeImgs/11_filter_roundish.png?raw=true' alt="Rounded Filter" width='300' />

Looking on the original images with the ruler I measured one centemetre to approimate 57 pixels in length. Each pebbles area is given in pixles, this is converted to cm^2 with some maths, the processed images are saved to `images/analysed` a plot graph is created too:

<img src='./images/avg_sizes_plot.png?raw=true' alt="Average Sizes" width='550' />

On the images with the larger pebbles I think the sample size and variety of sizes is confusing the results, the settings are the same for all images and are a bit of a compromise, tweaking the settings would probably spoil the measuements in other images. With all that, I can confirm that the pebbles do, indeed, get larger as you traverse the beach from North West to South East. 

## Other Options
These options are useful for fine-tuning the filters and inspecting the results. Image numbers are in the range 1 to 33 and correspond to those found in the `images/source/` or `images/cropped/` directories

__clickimage__ is very useful for quickly seeing the actual numbers used by the filters (see `clkImage.py`) clisk on a selected pebble to see some numbers.
```
python shrimpRocks.py --clickimage <image number>
```
__chug__ use this to output a range of images for a particular filter, some setup is needed in chugSegment in imgAnalyse.py. Output files are saved to images/chugtest, I used this to tune the default values for the filters in `imgFilters.py`. 
```
python shrimpRocks.py --chug <image number>
```
## Links and Sources

<a href='https://github.com/facebookresearch/segment-anything' target='_blank'>https://github.com/facebookresearch/segment-anything</a>
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```



