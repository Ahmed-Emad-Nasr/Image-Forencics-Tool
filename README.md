# Image Forensics GUI Application

A simple Python GUI application for image forensics using Tkinter.

## Features

- Load and display images
- Preprocessing: Convert to grayscale or YCbCr, resize
- Algorithms: GLCM, DCT, DWT, LBP
- Display results and logs

## Requirements

- Python 3.x
- Libraries: opencv-python, numpy, scikit-image, pywavelets, Pillow

## Installation

1. Create a virtual environment: `python -m venv .venv`
2. Activate: `.venv\Scripts\activate` (Windows)
3. Install packages: `pip install -r requirements.txt`

## Usage

Run: `python image_forensics_gui.py`

1. Click "Upload Image" to load an image.
2. Select preprocessing options.
3. Choose an algorithm.
4. Click "Process" to run.

## Code Structure

- `image_forensics_gui.py`: Contains the GUI class and display functions.
- `algorithms.py`: Contains image processing functions (preprocess, apply algorithms).