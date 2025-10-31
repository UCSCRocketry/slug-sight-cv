import os
import glob
import cv2
import numpy as np

CALIB_DIR = "calib_images_USE"   # Your calibration image directory
OUTPUT_DIR = "calib_output"
CHECKERBOARD = (8, 6)

# ---------------- Load Calibration Parameters ---------------- #
params = np.load(os.path.join(OUTPUT_DIR, "stereo_params.npz"))
K1, D1 = params["K1"], params["D1"]   # Left camera intrinsics + distortion
K2, D2 = params["K2"], params["D2"]   # Right camera intrinsics + distortion

print("Loaded camera intrinsics & distortion.")

def find_corners(gray):
    """ Returns detected checkerboard corners (refined) or (False, None). """
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


# ---------------- Load Image Pairs ---------------- #
left_images = sorted(glob.glob(os.path.join(CALIB_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(CALIB_DIR, "right_*.jpg")))
num_pairs = min(len(left_images), len(right_images))

if num_pairs == 0:
    raise RuntimeError("No matching left/right pairs found.")

print(f"Found {num_pairs} image pairs.\n")

# ---------------- Compute Homography From First Valid Pair ---------------- #
H = None
for lp, rp in zip(left_images, right_images):
    left = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)

    okL, cL = find_corners(left)
    okR, cR = find_corners(right)

    if okL and okR:
        ptsL = cL.reshape(-1, 2)
        ptsR = cR.reshape(-1, 2)

        # Compute Homography mapping RIGHT → LEFT frame
        H, mask = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 5.0)
        print(f"✅ Homography computed using: {os.path.basename(lp)}, {os.path.basename(rp)}")
        break

if H is None:
    raise RuntimeError("Could not compute homography — no usable corner pairs found.")

# Convert to LEFT → RIGHT warp
H = np.linalg.inv(H)

np.save(os.path.join(OUTPUT_DIR, "homography.npy"), H)
print("💾 Saved homography: calib_output/homography.npy\n")


# ---------------- Stitch All Pairs Using Homography ---------------- #
STITCH_DIR = os.path.join(OUTPUT_DIR, "stitched_pairs")
os.makedirs(STITCH_DIR, exist_ok=True)
print("Stitching all image pairs...\n")

for i, (lp, rp) in enumerate(zip(left_images, right_images)):
    left = cv2.imread(lp)
    right = cv2.imread(rp)

    if left is None or right is None:
        continue

    # Undistort using intrinsics
    left = cv2.undistort(left, K1, D1)
    right = cv2.undistort(right, K2, D2)

    hL, wL = left.shape[:2]
    hR, wR = right.shape[:2]

    # Project corners of LEFT image
    corners_left = np.array([[0,0],[wL,0],[wL,hL],[0,hL]], dtype=np.float32).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners_left, H)

    corners_right = np.array([[0,0],[wR,0],[wR,hR],[0,hR]], dtype=np.float32).reshape(-1,1,2)

    all_pts = np.vstack((projected, corners_right))

    x_min, y_min = np.int32(all_pts.min(axis=0).ravel())
    x_max, y_max = np.int32(all_pts.max(axis=0).ravel())

    # Translate so stitched image has no negative coordinates
    translation = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], np.float32)
    H_shifted = translation @ H

    canvas_w, canvas_h = x_max - x_min, y_max - y_min

    # Warp LEFT into RIGHT’s perspective
    stitched = cv2.warpPerspective(left, H_shifted, (canvas_w, canvas_h))

    # Paste RIGHT inside stitched canvas
    stitched[-y_min:hR-y_min, -x_min:wR-x_min] = right

    save_path = os.path.join(STITCH_DIR, f"stitched_{i:02d}.jpg")
    cv2.imwrite(save_path, stitched)
    print(f"Saved: {save_path}")

print("\nFinished stitching ALL image pairs.")
print(f"Output folder: {STITCH_DIR}\n")
print("Left image is warped into right frame → combined panoramic result.")
