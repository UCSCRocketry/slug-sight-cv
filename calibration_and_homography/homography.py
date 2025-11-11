import cv2 as cv
import numpy as np
import glob
import os

CALIB_DIR = "calib_images_USE"
CHESSBOARD_SIZE = (8, 6)

# Load left and right image sets
left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right_*.jpg")))

# Ensure we have pairs
if len(left_images) != len(right_images):
    print("Error: Left and right image counts do not match.")
    exit(1)

# termination criteria for subpixel corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Read one pair for testing (use first successful pair)
for left_path, right_path in zip(left_images, right_images):
    imgL = cv.imread(left_path)
    imgR = cv.imread(right_path)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    retR, cornersR = cv.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

    if retL and retR:
        # refine corner locations for accuracy
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # compute homography between the two sets of 2D points
        H, mask = cv.findHomography(cornersL, cornersR, cv.RANSAC)
        print("Homography Matrix (Left → Right):\n", H)

        # warp the left image to align with the right
        h, w, _ = imgR.shape
        warpedL = cv.warpPerspective(imgL, H, (w, h))

        # show side-by-side comparison
        combined = np.hstack((imgR, warpedL))
        break
print("Done.")