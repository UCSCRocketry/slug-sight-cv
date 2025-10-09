"""
calibrate_and_compute_homography.py
Author: Dane Pearson (danepearson on GitHub)

- Loads matched image pairs: images/cam0/cam0_*.jpg and images/cam1/cam1_*.jpg
- Detects chessboard corners
- Computes camera intrinsics for each camera
- Performs stereo calibration (R, T)
- Computes a planar homography H mapping cam1 -> cam0 using all detected corner correspondences
- Saves results to .npz files

Edit CHECKERBOARD, square_size_mm, and the image folder paths before running.
"""

import cv2
import numpy as np
import glob
import os

# -------- User-editable settings ----------
CHECKERBOARD = (8, 6)  # number of inner corners per row and column (for a 9x7 board)
square_size_mm = 25.0  # side length of one printed square in millimeters
cam0_pattern = "images/cam0/*.jpg"
cam1_pattern = "images/cam1/*.jpg"
output_npz = "calib_results.npz"
# -----------------------------------------

# Termination criteria for refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (same for all images) in world units (mm)
objp = np.zeros((CHECKERBOARD[1]*CHECKERBOARD[0], 3), np.float32)
# Note: mgrid order is (cols, rows) -> (x, y)
objp[:, :2] = (np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)) * square_size_mm

# Arrays to store object points and image points from all the images.
objpoints = []       # 3d points in real world space
imgpoints0 = []      # 2d points in cam0 image plane
imgpoints1 = []      # 2d points in cam1 image plane
image_shape = None

# Gather sorted file lists and ensure pairs match by index
files0 = sorted(glob.glob(cam0_pattern))
files1 = sorted(glob.glob(cam1_pattern))

if len(files0) != len(files1):
    raise SystemExit("Number of images in cam0 and cam1 folders differ. Use matched naming so pairs are aligned.")

if len(files0) < 6:
    raise SystemExit("Need at least ~10 good image pairs. Found {}".format(len(files0)))

print(f"Found {len(files0)} image pairs; pattern {CHECKERBOARD}; square size {square_size_mm} mm")

for f0, f1 in zip(files0, files1):
    img0 = cv2.imread(f0)
    img1 = cv2.imread(f1)
    if img0 is None or img1 is None:
        print("Skipping missing file pair:", f0, f1)
        continue

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if image_shape is None:
        image_shape = gray0.shape[::-1]  # (width, height)
    else:
        if gray0.shape != gray1.shape:
            print("Warning: image sizes differ for pair", f0, f1)
            continue

    # Find chessboard corners
    ret0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret0 and ret1:
        # refine corners
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints0.append(corners0)
        imgpoints1.append(corners1)
        print("Corners found for pair:", os.path.basename(f0), os.path.basename(f1))
    else:
        print("Chessboard not found on both images of pair, skipping:", f0, f1)

print(f"Total successful pairs: {len(objpoints)}")

# --------------- Single camera calibration ---------------
ret0, K0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints, imgpoints0, image_shape, None, None)
ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, image_shape, None, None)

print("Single-camera reproj error approx (cam0):", ret0)
print("Single-camera reproj error approx (cam1):", ret1)
print("K0:", K0)
print("K1:", K1)

# --------------- Stereo calibration ---------------
# We fix intrinsics (recommended if you already calibrated single cameras well). If you want to refine intrinsics too, remove the flag.
flags = cv2.CALIB_FIX_INTRINSIC

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, K0_ref, dist0_ref, K1_ref, dist1_ref, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints0, imgpoints1,
    K0, dist0, K1, dist1,
    image_shape,
    criteria=stereocalib_criteria,
    flags=flags
)
print("Stereo calibrate RMS:", ret_stereo)
print("R (rotation):\n", R)
print("T (translation):\n", T)

# --------------- Stereo rectify (useful later) ---------------
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K0_ref, dist0_ref, K1_ref, dist1_ref, image_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map0_x, map0_y = cv2.initUndistortRectifyMap(K0_ref, dist0_ref, R1, P1, image_shape, cv2.CV_32FC1)
map1_x, map1_y = cv2.initUndistortRectifyMap(K1_ref, dist1_ref, R2, P2, image_shape, cv2.CV_32FC1)

# --------------- Compute homography using all chessboard correspondences ---------------
# Stack all found corners into Nx2 arrays
pts0_all = np.vstack([p.reshape(-1, 2) for p in imgpoints0])
pts1_all = np.vstack([p.reshape(-1, 2) for p in imgpoints1])

H, mask = cv2.findHomography(pts1_all, pts0_all, cv2.RANSAC, 5.0)   # maps cam1 -> cam0
print("Homography H (cam1 -> cam0):\n", H)

# --------------- Evaluate homography reprojection error ---------------
pts1_transformed = cv2.perspectiveTransform(pts1_all.reshape(-1, 1, 2), H).reshape(-1, 2)
errors = np.linalg.norm(pts0_all - pts1_transformed, axis=1)
rmse = np.sqrt(np.mean(errors**2))
print("Homography RMSE (px):", rmse)
print("Sample reprojection errors (px):", errors[:10])

# --------------- Save results ---------------
np.savez(output_npz,
         K0=K0_ref, dist0=dist0_ref, K1=K1_ref, dist1=dist1_ref,
         R=R, T=T, E=E, F=F, H=H, image_shape=image_shape, CHECKERBOARD=CHECKERBOARD, square_size_mm=square_size_mm)
print("Saved calibration to", output_npz)