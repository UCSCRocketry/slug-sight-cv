import numpy as np
import cv2 as cv
import os
import glob

CALIB_DIR = "calib_images_USE"

#termination criteria, maximum iterations and minimum improvement(epsilon)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#This is the size of each square on the checkerboard
#Each square right now is 0.27 meters
square_size = 0.27
#This is the dimensions of the checkerboard
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) * square_size

# Arrays to store object points and image points from all the images.
L_objpoints = [] # 3d point in real world space
L_imgpoints = [] # 2d points in image plane.
R_objpoints = [] # 3d point in real world space
R_imgpoints = [] # 2d points in image plane.

lshape = None
rshape = None
left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right*.jpg")))
for limage, rimage in zip(left_images, right_images):
    img = cv.imread(limage)
    left = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(left, (7,6), None)
    lshape = np.shape(left)
    # If found, add object points, image points (after refining them)
    if ret == True:
        L_objpoints.append(objp)
        corners2 = cv.cornerSubPix(left,corners, (11,11), (-1,-1), criteria)
        L_imgpoints.append(corners2)
    
    img = cv.imread(rimage)
    right = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(right, (7,6), None)
    rshape = np.shape(right)
    # If found, add object points, image points (after refining them)
    if ret == True:
        R_objpoints.append(objp)
        corners2 = cv.cornerSubPix(right ,corners, (11,11), (-1,-1), criteria)
        R_imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(L_objpoints, L_imgpoints, lshape[::-1], None, None)
print(f"Left Calibration: ret = {ret}\n mtx = \n{mtx}\n rvecs = \n{rvecs}\n tvces = \n{tvecs}\n")
# Arrays to store object points and image points from all the images.
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(R_objpoints, R_imgpoints, rshape[::-1], None, None)
print(f"Right Calibration: ret = {ret}\n mtx = \n{mtx}\n rvecs = \n{rvecs}\n tvces = \n{tvecs}\n")