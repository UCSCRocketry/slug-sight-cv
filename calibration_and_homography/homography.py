import time
import cv2
import numpy as np

# Camera indexes (adjust if needed)
LEFT_CAM_INDEX  = 0
RIGHT_CAM_INDEX = 1

# Load stereo calibration results
params = np.load("calib_output/stereo_params.npz")
K1, D1 = params["K1"], params["D1"]
K2, D2 = params["K2"], params["D2"]

# Downscale factor — lower = faster
SCALE = 0.5   # 0.5 → half resolution (try 0.4 or 0.33 for Pi Zero class boards)

# FAST corner detector (quick keypoint finder)
FAST = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)

# BRIEF descriptor extractor (very fast, pairs well with FAST)
BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Hamming-based matcher (required for BRIEF descriptors)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# RANSAC threshold for homography
RANSAC_THRESH = 3.0

# Optional display
SHOW_WINDOW = True


def make_undistort_maps(K, D, size):
    """Precompute undistortion remap grids for per-frame fast correction."""
    w, h = size
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    return cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)


def detect_and_describe(gray):
    """FAST → BRIEF pipeline."""
    keypoints = FAST.detect(gray, None)
    if not keypoints:
        return None, None
    keypoints, descriptors = BRIEF.compute(gray, keypoints)
    return keypoints, descriptors


def main():
    capL = cv2.VideoCapture(LEFT_CAM_INDEX)
    capR = cv2.VideoCapture(RIGHT_CAM_INDEX)

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Could not open both cameras.")

    # Get frame size
    okL, frameL = capL.read()
    okR, frameR = capR.read()
    if not okL or not okR:
        raise RuntimeError("Failed to read initial frames.")

    h, w = frameL.shape[:2]
    size = (w, h)

    # Precompute remap maps for undistortion
    map1_L, map2_L = make_undistort_maps(K1, D1, size)
    map1_R, map2_R = make_undistort_maps(K2, D2, size)

    print("✅ Running optimized homography on Raspberry Pi...\n")
    fps_timer = time.time()
    frames = 0

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            continue

        # Undistort
        undistL = cv2.remap(frameL, map1_L, map2_L, cv2.INTER_LINEAR)
        undistR = cv2.remap(frameR, map1_R, map2_R, cv2.INTER_LINEAR)

        # Downscale for speed
        smallL = cv2.resize(undistL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        smallR = cv2.resize(undistR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale
        grayL = cv2.cvtColor(smallL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(smallR, cv2.COLOR_BGR2GRAY)

        # FAST + BRIEF feature detection/description
        kL, dL = detect_and_describe(grayL)
        kR, dR = detect_and_describe(grayR)
        if dL is None or dR is None:
            continue

        # Match descriptors
        matches = BF.match(dL, dR)
        if len(matches) < 12:
            continue

        # Extract matched points
        ptsL = np.float32([kL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ptsR = np.float32([kR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, RANSAC_THRESH)

        # FPS display
        frames += 1
        if frames % 10 == 0:
            dt = time.time() - fps_timer
            print(f"FPS: {frames/dt:.1f}  matched: {len(matches)}")
            fps_timer = time.time()
            frames = 0

        # Optional visualization
        if SHOW_WINDOW and H is not None:
            disp = cv2.drawMatches(smallL, kL, smallR, kR, matches[:30], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Homography (FAST+BRIEF, Pi Optimized)", disp)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    capL.release()
    capR.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
