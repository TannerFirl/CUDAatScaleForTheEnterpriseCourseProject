## Project Description

This project overlays an rgba image on an rgb image using alpha blending. The alpha blending is implemented as a cuda kernel. For information on what alpha blending is and why it is useful, click [here](https://learnopencv.com/alpha-blending-using-opencv-cpp-python/).

# CUDAatScaleForTheEnterpriseCourseProject Quickstart

To run this, you must have access to an nvidia gpu and software. As such, you may wish to use a lab environment from the ["CUDA at Scale for the Enterprise" Coursera course](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise/home/welcome) to execute it. If you have not already, clone this repository, cd to its base directory, and initialize git submodules:

```
git clone https://github.com/TannerFirl/CUDAatScaleForTheEnterpriseCourseProject.git
cd CUDAatScaleForTheEnterpriseCourseProject
git submodule --init --recursive
```

## Overlay included sample images (test setup)

```
git submodule --init --recursive
make clean build run
```

output.png will contain the sample images overlayed using alpha blending.

## Overlay custom images

NOTE: the watermark MUST be an RGBA transparent .png file. Not all .png files are RGBA transparent. I have had luck using [this tool to convert images to RGBA transparent .png files](https://fconvert.com/image/convert-to-png/). Select `Depth color: 32 (True color, RGBA, transparent)` and download the converted image. For more information regarding types of .png files, click [here](http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.8).

```
git submodule --init --recursive
make clean build
./bin/watermark <8bit-rgb-img-filename.png> <8bit-rgba-watermark-filename.png> [output.png]
```

## Code Organization

```bin/```
This folder holds all binary/executable code that is built automatically or manually.

```data/```
This folder holds all example data in any format.

```src/```
The source code for this project.

```README.md```
The description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```Makefile```
Script for building this project's code

```third-party/cuda-samples```
cuda's samples repository, which is a submodule of this repository.
