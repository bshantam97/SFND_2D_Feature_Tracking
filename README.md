## Author: Shantam Bajpai

# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

# Performance Evaluations and Description of each part of the project 

## MP.1 Data Buffer Optimization
Now for this project a ring buffer has been implemented. The idea behind the ring buffer implementation is that at anytime there should not be more than 2 images in the databuffer. Hence when an image is pushed back into the data buffer vector and if the size is greater than 2 the image at the initial position of the vector is removed.

## MP.2 Keypoint Detection
In this project a variety of keypoint detecetors have been implemented using OpenCV. The keypoint detectors that have been implemented are SHI-TOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT and FREAK.

## MP.3 Keypoint Removal
We are only concerned with the keypoints that have been detected on the vehicle. The keypoints describe a coordinate of interest like a corner or an edge which is then used to by the descriptors to define a local region of interest around the keypoint. The keypoints that were outside the defined box region were discarded and only the ones inside the box region containing the car were kept.

## MP.4 Keypoint Descriptors
A variety of keypoint descriptors were implemented like BRISK, SIFT, ORB, BRIEF and FREAK. All these descriptors were used in combination with the keypoint detectors to select the best performing system for 2d tracking.

## MP.5 Descriptor Matching
Once we have detected descriptors of all the images we need to perform descriptor matching. This can be done using Brute Force matching in which for each descriptor in the source image we compare and match it with all the descriptors in the reference image with and without **cross check matching**. Th other approach and a much faster approach is to use FLANN based matching which has also been implemented. Then either using **Nearest Neighbors** we find the best descriptor matches or using **K-Nearest neighbors** we find the k-best descriptor matches and then based on the **Distance Ratio Test" refine our descriptor matches.

## MP.6 Descriptor Ratio Test
Using the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

## MP.7 Performance Evaluation 1 (Number of Keypoints Detected)

| Keypoint Detector | Image 1 | Image 2 | Image 3 | Image 4 | Image 5 | Image 6 | Image 7 | Image 8 | Image 9 | Image 10 | Neighborhood size |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| SHI-TOMASI | 127 | 120 | 123 | 120 | 120 | 115 | 114 | 125 | 112 | 113 | 128x50 | 
| HARRIS | 20 | 17 | 21 | 21 | 35 | 25 | 20 | N/A | N/A | N/A | 128x35 |
| FAST | 95 | 95 | 93 | 98 | 98 | 99 | 90 | 92 | 101 | 95 | 128x50 |
| BRISK | 254 | 274 | 276 | 275 | 293 | 275 | 289 | 268 | 260 | 250 | 128x50 |
| ORB | 91 | 102 | 106 | 113 | 109 | 124 | 129 | 127 | 124 | 125 | 128x50 |
| AKAZE | 162 | 157 | 159 | 154 | 162 | 163 | 173 | 175 | 175 | 175 | 128x50 |
| SIFT | 137 | 131 | 121 | 135 | 134 | 139 | 136 | 147 | 156 | 135 | 128x51 |

## MP.8 Performance Evaluation 2 (Number of matched keypoints between a set of images)

| Keypoint Detector/Descriptor | BRISK | SIFT | ORB | BRIEF | FREAK | AKAZE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| SHI-TOMASI | 349 | 406 | 385 | 401 | 299 | N/A |
| HARRIS | 10 | 70 | 31 | 53 | 22 | N/A |
| FAST | 303 | 339 | 362 | 369 | 289 | N/A |
| BRISK | 274 | 278 | 260 | 281 | 280 | N/A |
| ORB | 312 | 298 | 299 | 217 | 221 | N/A |
| AKAZE | N/A | N/A | N/A | N/A | N/A | 342 |
| SIFT | 159 | 297 | N/A | 203 | 174 | N/A |

## MP.9 Performance Evaluation 3 (Upper limit Time for Keypoint Detection and description extraction)

| Keypoint Detector/Descriptor | BRISK | SIFT | ORB | BRIEF | FREAK | AKAZE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| SHI-TOMASI | 22.5156ms | 33ms | 20.5ms | 17ms | 46ms | N/A |
| HARRIS | N/A | N/A | N/A | N/A | N/A | N/A |
| FAST | 1ms | 20ms  | 6ms | 3ms | 36ms | N/A |
| BRISK | 55ms | 78ms | 130ms | 66ms | 92ms | N/A |
| ORB | 10ms | 49ms | 26ms | 9ms | 49ms | N/A |
| AKAZE | N/A | N/A | N/A | N/A | N/A | 212ms |
| SIFT | 114ms | 207ms | N/A | 104ms | 162ms | N/A |

After looking at the above performance evaluations I have listed the top 3 detector-descriptor combinations for keypoint detection and matching between vehicles
1. FAST Detector with BRIEF Descriptor
2. FAST Detector with BRISK Descriptor
3. AKAZE Detector and descriptor (might be slow but provides high quality correspondences_
## Dependencies for Running Locally
1. cmake >= 2.8
 * All OSes: [click here for installation instructions](https://cmake.org/install/)

2. make >= 4.1 (Linux, Mac), 3.81 (Windows)
 * Linux: make is installed by default on most Linux distros
 * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
 * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)

3. OpenCV >= 4.1
 * All OSes: refer to the [official instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
 * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors. If using [homebrew](https://brew.sh/): `$> brew install --build-from-source opencv` will install required dependencies and compile opencv with the `opencv_contrib` module by default (no need to set `-DOPENCV_ENABLE_NONFREE=ON` manually). 
 * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)

4. gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using either [MinGW-w64](http://mingw-w64.org/doku.php/start) or [Microsoft's VCPKG, a C++ package manager](https://docs.microsoft.com/en-us/cpp/build/install-vcpkg?view=msvc-160&tabs=windows). VCPKG maintains its own binary distributions of OpenCV and many other packages. To see what packages are available, type `vcpkg search` at the command prompt. For example, once you've _VCPKG_ installed, you can install _OpenCV 4.1_ with the command:
```bash
c:\vcpkg> vcpkg install opencv4[nonfree,contrib]:x64-windows
```
Then, add *C:\vcpkg\installed\x64-windows\bin* and *C:\vcpkg\installed\x64-windows\debug\bin* to your user's _PATH_ variable. Also, set the _CMake Toolchain File_ to *c:\vcpkg\scripts\buildsystems\vcpkg.cmake*.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.
