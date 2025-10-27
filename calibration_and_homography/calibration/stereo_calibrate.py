# https://github.com/erencanbulut/rpi_stereo_cam_calibration
import glob
import os

import cv2
import numpy as np

# === CONFIGURATION ===
chessboard_size = (9, 6)  # inner corners (width, height)
square_size = 0.04  # in meters (40mm)

# === Paths ===
calib_dir = "calib"
left_images = sorted(glob.glob(os.path.join(calib_dir, "left_*.png")))
right_images = sorted(glob.glob(os.path.join(calib_dir, "right_*.png")))

# === Prepare object points ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left images
imgpoints_right = []  # 2D points in right images

# === Detect corners ===
for left_img_path, right_img_path in zip(left_images, right_images):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
    else:
        print(f"Skipping pair: {left_img_path}, {right_img_path}")

# === Calibration ===
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=flags
)

# === Rectification ===
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    grayL.shape[::-1], R, T, alpha=0
)

# === Save Calibration Data ===
np.savez("stereo_calib.npz",
         cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
         R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("\n Calibration complete. Saved as stereo_calib.npz")
