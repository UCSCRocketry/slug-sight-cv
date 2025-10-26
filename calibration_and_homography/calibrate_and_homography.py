import cv2
import numpy as np
import glob
import os

# Path from calibrate-image_pairs.py
CALIB_DIR = "calib_images"

# (7, 6) means 7 inner corners horizontally, 6 vertically
CHECKERBOARD = (7, 6)
SQUARE_SIZE = 0.27  # measured dimensions

# Termination criteria for corner refinement
# Stops either after 30 iterations or when corner position accuracy improves < 0.001
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Loading Image Pairs from capture_images.py and sort
left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right_*.jpg")))

# Check for list lengths
if len(left_images) != len(right_images):
    print(f"Warning: {len(left_images)} left and {len(right_images)} right images found. Using the minimum.")

# Use minimum number of pairs that exist
num_pairs = min(len(left_images), len(right_images))

if num_pairs == 0:
    raise RuntimeError("No image pairs found. Make sure capture_images.py ran successfully.")

print(f"Found {num_pairs} image pairs for calibration.\n")

# Object points represent the 3D coordinates of each checkerboard corner
# Builds an array with one row per checkerboard inner corner; each row has three columns (x, y, z)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # scale from units to meters

# Lists to store all successful detections
objpoints = []  # 3D points in real world space
imgpoints_L = []  # 2D points in left camera (pixel points)
imgpoints_R = []  # 2D points in right camera (pixel points)

image_size = None
first_good_pair = None  # for homography computation


# Find Chessboard Corners
for i in range(num_pairs):
    left = cv2.imread(left_images[i])
    right = cv2.imread(right_images[i])

    if left is None or right is None:
        print(f"Could not read pair {i}, skipping.")
        continue

    # findChessboardCorners requires grayscale cconversion
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Store image size once
    if image_size is None:
        image_size = (grayL.shape[1], grayL.shape[0])  # (width, height) (we reshaped this earlier)

    # Try to find the inner corners of the checkerboard
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if foundL and foundR:
        # Refine corner positions for higher accuracy
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), CRITERIA)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), CRITERIA)

        # Store points
        objpoints.append(objp)
        imgpoints_L.append(cornersL)
        imgpoints_R.append(cornersR)

        if first_good_pair is None:
            first_good_pair = (cornersL.copy(), cornersR.copy())

        print(f"✅ Pair {i}: corners found.")
    else:
        print(f"⚠️  Pair {i}: corners NOT found.")

print(f"\nTotal valid pairs used: {len(objpoints)}")
if len(objpoints) < 5:
    raise RuntimeError("Not enough valid pairs for accurate calibration (need at least ~10).")


# Calibrate LEFT camera
retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_L, image_size, None, None
)

# Calibrate RIGHT camera
retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_R, image_size, None, None
)

print("\n🎯 Intrinsic Calibration Results:")
print("Left RMS error:", retL)
print("Right RMS error:", retR)


# Calibration - Finds the rotation (R) and translation (T) between the two cameras

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
flags = cv2.CALIB_FIX_INTRINSIC

retStereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R,
    K1, D1, K2, D2, image_size,
    criteria=stereo_criteria, flags=flags
)

print("\n🔧 Stereo Calibration Complete")
print(f"Reprojection Error: {retStereo}")
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)


# Homography calculation (Left to Right)
if first_good_pair is not None:
    ptsL = first_good_pair[0].reshape(-1, 2)
    ptsR = first_good_pair[1].reshape(-1, 2)
    H, mask = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, 5.0)
    print("\n Homography Matrix (H):\n", H)
else:
    raise RuntimeError("No valid pair found for homography computation.")

# Save Calibration in calib_output as npy and npz
os.makedirs("calib_output", exist_ok=True)

np.savez("calib_output/stereo_params.npz",
         K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
         image_width=image_size[0], image_height=image_size[1],
         checkerboard_cols=CHECKERBOARD[0],
         checkerboard_rows=CHECKERBOARD[1],
         square_size=SQUARE_SIZE)

np.save("calib_output/homography.npy", H)

print("\n Saved calibration results:")
print(" - calib_output/stereo_params.npz  (intrinsics, distortion, R, T)")
print(" - calib_output/homography.npy     (3x3 homography matrix)\n")

print("Calibration + Homography Complete!")
