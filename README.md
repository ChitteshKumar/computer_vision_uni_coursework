# computer_vision_uni_coursework

This repository contains three completed courseworks from the Computer Vision course at University of Manchester. Each coursework demonstrates different techniques and methodologies in computer vision, providing a comprehensive overview of the concepts learned throughout the course.

## Courseworks Overview

### Coursework 1: Image Smoothing and Edge Detection
- **Objective:** Apply smoothing filters and edge detection techniques.
- **Key Techniques:**
  - Average (Mean) Smoothing Filter
  - Weighted-Average (Gaussian) Smoothing Filter
  - Sobel Operator for Edge Detection
  - Thresholding for Object Segmentation
    
### Coursework 2: Object Detection
- **Objective:** Detect objects in images using feature detection and matching.
- **Key Techniques:**
  - Harris Corner Detection
      - Feature Detection
      - Implementation from Scratch
  - ORB (Oriented FAST and Rotated BRIEF) Feature Detection and Description
      - Built-in ORB Framework
      - Keypoint Detection
  - Feature Matching using Sum-of-Squared Differences (SSD)
      - Implementation using SciPy
      - Distance Calculation
  - Ratio Test for Vague Matches
      - Euclidean Distance
      - Thresholding for Valid Matches

### Coursework 3: Stereo Imagery and Selective Focus
- **Objective:** Handle stereo imagery to create 3D models and apply selective focus.
- **Key Techniques:**
  - Focal Length Calculation
      - Formula Application
      - Parameter Estimation
  - Disparity Map Creation using Canny Edge Detection
      - Edge Detection with Canny
      - Disparity Calculation
  - 3D Scene Reconstruction from Disparity Data
      - Coordinate Calculation
      - 3D Visualization
  - Selective Focus based on Depth Information
      - Depth Image Creation
      - Foreground and Background Segmentation

## Prerequisites
- Python 3.x
- NumPy
- OpenCV
- Matplotlib (for Coursework 3)
