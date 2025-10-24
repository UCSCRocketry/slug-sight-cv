import numpy as np
import cv2 as cv

for i in range(30):
    left = cv.imread(f"calib_images/left_{i:02d}.jpg")
    right = cv.imread(f"calib_images/right_{i:02d}.jpg") 