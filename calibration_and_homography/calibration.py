import os
import glob
import cv2
import numpy as np

# Path from capture_image_pairs.py
CALIB_DIR = "calib_images"

# (8, 6) means 8 inner corners horizontally, 6 vertically (9 x 7 squares)
CHECKERBOARD = (8, 6)

# Size of each square (meters)
SQUARE_SIZE = 0.27

# Termination criteria for subpixel corner refinement
# Stops either after 30 iterations or when corner position accuracy improves < 0.001
CRITERIA_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Termination criteria for stereo optimization
STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Stereo flag -> keep intrinsics fixed while solving R,T,E,F
STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC

def find_corners(gray, pattern):
    """
    Finds inner corners on a grayscale image.
    - Uses findChessboardCornersSB if available (more robust).
    - Falls back to findChessboardCorners + cornerSubPix.

    Returns:
        found (bool), corners (np.ndarray shape (N,1,2) float32)
    """
    # Try SB (single-board) detector if present in current OpenCV build
    if hasattr(cv2, "findChessboardCornersSB"):
        found = cv2.findChessboardCornersSB(gray, pattern)
        if isinstance(found, tuple):
            # Newer OpenCV returns (bool, corners)
            ok, pts = found
        else:
            # Older variants may return only corners or None
            ok, pts = (found is not None), found
        if ok and pts is not None and len(pts) == pattern[0] * pattern[1]:
            return True, pts.astype(np.float32).reshape(-1, 1, 2)

    # Fallback: classic detector + subpixel refinement
    ok, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ok:
        return False, None

    # Refine to subpixel accuracy for better calibration quality
    refined = cv2.cornerSubPix(
        gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=CRITERIA_SUBPIX
    )
    return True, refined


def main():
    # Loading image lists and sorting to pair left_i with right_i
    left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
    right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right_*.jpg")))

    # Check for list length mismatch and bound by min length
    if len(left_images) != len(right_images):
        print(f"Warning: {len(left_images)} left and {len(right_images)} right images found. Using the minimum.")
    num_pairs = min(len(left_images), len(right_images))
    if num_pairs == 0:
        raise RuntimeError("No image pairs found. Make sure capture_image_pairs.py ran successfully.")
    print(f"Found {num_pairs} image pairs for calibration.\n")

    # Object points represent the 3D coordinates of each checkerboard inner corner on Z=0 plane
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # scale from grid units to meters

    # Arrays to accumulate per-view correspondences
    objpoints = []   # 3D points in real world space (list of (N,3))
    imgpoints_L = [] # 2D points in left images (list of (N,1,2))
    imgpoints_R = [] # 2D points in right images (list of (N,1,2))
    image_size = None

    # Find corners in all pairs
    successful = 0
    for i, (lp, rp) in enumerate(zip(left_images[:num_pairs], right_images[:num_pairs])):
        # Read grayscale directly for detector speed and correctness
        grayL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        grayR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if grayL is None or grayR is None:
            print(f"Could not read pair {i}, skipping.")
            continue

        # Store size once (width, height)
        if image_size is None:
            image_size = (grayL.shape[1], grayL.shape[0])

        # Detect corners in both images
        okL, cornersL = find_corners(grayL, CHECKERBOARD)
        okR, cornersR = find_corners(grayR, CHECKERBOARD)

        if okL and okR:
            objpoints.append(objp)
            imgpoints_L.append(cornersL)
            imgpoints_R.append(cornersR)
            successful += 1
            print(f"Pair {i}: corners found.")
        else:
            print(f"Pair {i}: corners NOT found.")

    # Sanity check on total successful pairs
    print(f"\nTotal valid pairs used: {successful}")
    if successful < 5:
        raise RuntimeError("Not enough valid pairs for accurate calibration (need at least ~10).")

    # Calibrate LEFT camera intrinsics + distortion
    retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpoints_L, image_size, None, None
    )

    # Calibrate RIGHT camera intrinsics + distortion
    retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpoints_R, image_size, None, None
    )

    # Print RMS reprojection errors for intrinsics
    print("\n Intrinsic Calibration Results:")
    print("Left RMS error:", retL)
    print("Right RMS error:", retR)

    # Stereo calibration to find relative pose R, T (keeping intrinsics fixed)
    retStereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K1, D1, K2, D2, image_size,
        criteria=STEREO_CRITERIA, flags=STEREO_FLAGS
    )

    # Print stereo reprojection error and pose
    print("\n Stereo Calibration Complete")
    print(f"Reprojection Error: {retStereo}")
    print("Rotation matrix (R):\n", R)
    print("Translation vector (T):\n", T)

    # Save parameters to disk for post-launch usage
    os.makedirs("calib_output", exist_ok=True)
    np.savez("calib_output/stereo_params.npz",
             K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
             image_width=image_size[0], image_height=image_size[1],
             checkerboard_cols=CHECKERBOARD[0],
             checkerboard_rows=CHECKERBOARD[1],
             square_size=SQUARE_SIZE)

    print("\n Saved calibration results:")
    print(" - calib_output/stereo_params.npz  (intrinsics, distortion, R, T)")
    print("\n Calibration Complete (pre-launch).")


if __name__ == "__main__":
    main()
