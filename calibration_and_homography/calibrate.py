import numpy as np
import cv2 as cv

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

for i in range(30):
    left = cv.imread(f"calib_images/left_{i:02d}.jpg") 
    ret, corners = cv.findChessboardCorners(left, (7,6), None)
    #ret, corners = cv.findChessboardCorners(right, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        L_objpoints.append(objp)
        corners2 = cv.cornerSubPix(left,corners, (11,11), (-1,-1), criteria)
        L_imgpoints.append(corners2)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(L_objpoints, L_imgpoints, left.shape[::-1], None, None)

# Arrays to store object points and image points from all the images.
R_objpoints = [] # 3d point in real world space
R_imgpoints = [] # 2d points in image plane.

for i in range(30):
    right = cv.imread(f"calib_images/right_{i:02d}.jpg") 
    ret, corners = cv.findChessboardCorners(left, (7,6), None)
    #ret, corners = cv.findChessboardCorners(right, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        R_objpoints.append(objp)
        corners2 = cv.cornerSubPix(right,corners, (11,11), (-1,-1), criteria)
        R_imgpoints.append(corners2)
        
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(R_objpoints, R_imgpoints, right.shape[::-1], None, None)