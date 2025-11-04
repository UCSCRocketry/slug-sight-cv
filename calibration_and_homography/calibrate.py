import numpy as np
import cv2 as cv
import os
import glob

CALIB_DIR = "calib_images_USE"

#termination criteria, maximum iterations and minimum improvement(epsilon)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#This is the size of each square on the checkerboard
#Each square right now is 0.27 meters
square_size = 0.027
#This is the dimensions of the checkerboard
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * square_size

# Arrays to store object points and image points from all the images.
L_objpoints = [] # 3d point in real world space
L_imgpoints = [] # 2d points in image plane.
R_objpoints = [] # 3d point in real world space
R_imgpoints = [] # 2d points in image plane.

lshape = None
rshape = None
left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right*.jpg")))
i = 1
for limage, rimage in zip(left_images, right_images):
    img = cv.imread(limage)
    left = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(left, (8,6), None)
    lshape = np.shape(left)
    # If found, add object points, image points (after refining them)
    if ret == True:
        L_objpoints.append(objp)
        corners2 = cv.cornerSubPix(left,corners, (11,11), (-1,-1), criteria)
        L_imgpoints.append(corners2)

    img = cv.imread(rimage)
    right = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(right, (8,6), None)
    rshape = np.shape(right)
    # If found, add object points, image points (after refining them)
    if ret == True:
        R_objpoints.append(objp)
        corners2 = cv.cornerSubPix(right ,corners, (11,11), (-1,-1), criteria)
        R_imgpoints.append(corners2)
    i+= 1
#root projection error, camera matrix, dist coefficients, rotation vector, translation vector
lret, lmtx, ldist, lrvecs, ltvecs = cv.calibrateCamera(L_objpoints, L_imgpoints, lshape[::-1], None, None)
#print(f"Left Calibration: ret = {lret}\n mtx = \n{lmtx}\n rvecs = \n{lrvecs}\n tvces = \n{ltvecs}\n")
# Arrays to store object points and image points from all the images.
rret, rmtx, rdist, rrvecs, rtvecs = cv.calibrateCamera(R_objpoints, R_imgpoints, rshape[::-1], None, None)
#print(f"Right Calibration: ret = {rret}\n mtx = \n{rmtx}\n rvecs = \n{rrvecs}\n tvces = \n{rtvecs}\n")

np.savez("calibration_data.npz", lret = lret, lmtx = lmtx, ldist=ldist, lrvecs = ltvecs, rret = rret, rmtx = rmtx, rdist = rdist, rrvecs = rrvecs, rtvecs = rtvecs)

H,mask = cv.findHomography(L_imgpoints[0], R_imgpoints[0])
np.save("homography.npy", H)

#todo
#GStreamer pipeline simulation
