import cv2
import numpy as np
import glob
import os

CALIB_DIR = "calib_images"

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.27

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right_*.jpg")))

if len(left_images) != len(right_images):
    print(f"Warning: {len(left_images)} left and {len(right_images)} right images found. Using the minimum.")

num_pairs = min(len(left_images), len(right_images))
if num_pairs == 0:
    raise RuntimeError("No image pairs found. Make sure capture_images.py ran successfully.")
print(f"Found {num_pairs} image pairs for calibration.\n")

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_L = []
imgpoints_R = []
image_size = None

for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left is None or right is None:
        print(f"Could not read pair {i}, skipping.")
        continue

    if image_size is None:
        image_size = (left.shape[1], left.shape[0])

    foundL, cornersL = cv2.findChessboardCorners(left, CHECKERBOARD, None)
    foundR, cornersR = cv2.findChessboardCorners(right, CHECKERBOARD, None)

    if foundL and foundR:
        cornersL = cv2.cornerSubPix(left, cornersL, (11, 11), (-1, -1), CRITERIA)
        cornersR = cv2.cornerSubPix(right, cornersR, (11, 11), (-1, -1), CRITERIA)

        objpoints.append(objp)
        imgpoints_L.append(cornersL)
        imgpoints_R.append(cornersR)

        print(f"Pair {i}: corners found.")
    else:
        print(f"Pair {i}: corners NOT found.")

print(f"\nTotal valid pairs used: {len(objpoints)}")
if len(objpoints) < 5:
    raise RuntimeError("Not enough valid pairs for accurate calibration (need at least ~10).")

retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_L, image_size, None, None
)

retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_R, image_size, None, None
)

print("\n Intrinsic Calibration Results:")
print("Left RMS error:", retL)
print("Right RMS error:", retR)

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
flags = cv2.CALIB_FIX_INTRINSIC

retStereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R,
    K1, D1, K2, D2, image_size,
    criteria=stereo_criteria, flags=flags
)

print("\n Stereo Calibration Complete")
print(f"Reprojection Error: {retStereo}")
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)

src_points = np.vstack(imgpoints_L).reshape(-1, 2)
dst_points = np.vstack(imgpoints_R).reshape(-1, 2)
H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
print("\n Homography Matrix (H):\n", H)

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

print("\n Calibration + Homography Complete!")
