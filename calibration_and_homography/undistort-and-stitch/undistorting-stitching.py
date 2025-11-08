import cv2
import numpy as np
import glob
import os
from collections import defaultdict
import re


def parse_image_files(folder_path):
    """Parse image files and organize them by camera (left/right)"""
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    left_images = []
    right_images = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = re.search(r'(left|right)_(\d+)_(\d{8})_', filename)
        if match:
            side = match.group(1)
            if side == 'left':
                left_images.append(img_path)
            else:
                right_images.append(img_path)
    
    return sorted(left_images), sorted(right_images)


def calibrate_single_camera(image_paths, chessboard_size=(6, 8), square_size=27.0):
    """Calibrate a single camera using its images"""
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    
    img_shape = None
    successful_images = []
        
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            successful_images.append(img_path)
        else:
            print(f"(Note: No chessboard in {os.path.basename(img_path)})\n")
    
    if len(objpoints) < 10:
        print(f"Warning: Only found {len(objpoints)} valid images. Need at least 10 for good calibration.")
        return None
        
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("Calibration failed.")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    return {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'img_shape': img_shape,
        'mean_error': mean_error,
        'successful_images': successful_images
    }


def create_undistortion_maps(calib_data, alpha=0.0):
    """Create undistortion maps from calibration data
    
    alpha=0: returns undistorted image with minimum unwanted pixels (crops more)
    alpha=1: retains all source image pixels (may have black borders)
    """
    mtx = calib_data['camera_matrix']
    dist = calib_data['dist_coeffs']
    img_shape = calib_data['img_shape']
    
    # Get optimal new camera matrix
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, alpha, img_shape)
    # Create undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, img_shape, cv2.CV_16SC2)
    
    return map1, map2, new_mtx, roi


def undistort_image(img, map1, map2):
    """Apply undistortion to an image"""
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def match_image_pairs(left_images, right_images):
    """Match left and right images by their pair number and timestamp"""
    pairs = []
    
    for left_path in left_images:
        left_filename = os.path.basename(left_path)
        match = re.search(r'left_(\d+)_(\d{8})_', left_filename)
        if match:
            pair_num, date = match.groups()
            # Find corresponding right image
            for right_path in right_images:
                right_filename = os.path.basename(right_path)
                if f'right_{pair_num}_{date}_' in right_filename:
                    pairs.append((left_path, right_path))
                    break
    
    return sorted(pairs)


def compute_homography_from_undistorted(left_images, right_images, left_map1, left_map2, 
                                        right_map1, right_map2, num_pairs=10):
    """Compute homography between undistorted left and right images
    Maps RIGHT image to LEFT image coordinate system"""
    
    pairs = match_image_pairs(left_images, right_images)
    pairs = pairs[:num_pairs]  # Use first N pairs
    
    all_src_pts = []
    all_dst_pts = []
        
    for left_path, right_path in pairs:
        # Load and undistort images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        left_undist = undistort_image(left_img, left_map1, left_map2)
        right_undist = undistort_image(right_img, right_map1, right_map2)
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(right_gray, None)  # RIGHT is source
        kp2, des2 = orb.detectAndCompute(left_gray, None)   # LEFT is destination
        
        if des1 is None or des2 is None:
            continue
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches
        good_matches = matches[:100]
        
        if len(good_matches) < 10:
            continue
        
        # Extract matched points - RIGHT to LEFT
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])  # RIGHT
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # LEFT
        
        all_src_pts.extend(src_pts)
        all_dst_pts.extend(dst_pts)
            
    if len(all_src_pts) < 20:
        print("Not enough matches found.")
        return None
    
    # Compute homography from all collected points
    all_src_pts = np.array(all_src_pts)
    all_dst_pts = np.array(all_dst_pts)
    
    # H maps RIGHT image to LEFT image coordinate system
    H, mask = cv2.findHomography(all_src_pts, all_dst_pts, cv2.RANSAC, 5.0)
    print(f"(Homography computed from {len(all_src_pts)} total feature matches)")
    
    return H


def stitch_undistorted_pair(left_path, right_path, left_map1, left_map2, 
                            right_map1, right_map2, homography, output_path):
    """Stitch a pair of undistorted images with simple cut (no blending)
    Note: The 'left' and 'right' labels are swapped - left files are actually right camera"""
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    # Undistort
    left_undist = undistort_image(left_img, left_map1, left_map2)
    right_undist = undistort_image(right_img, right_map1, right_map2)
    
    # SWAP: "left" files are actually the right camera view, and vice versa
    actual_left = right_undist  # right_undist is the actual left view
    actual_right = left_undist  # left_undist is the actual right view
    
    h, w = actual_left.shape[:2]
    
    # Try smaller overlap - maybe 15-20%
    overlap_pixels = int(w * 0.15)  # 15% overlap
    
    # Create canvas that accounts for overlap
    canvas_width = w * 2 - overlap_pixels
    result = np.zeros((h, canvas_width, 3), dtype=np.uint8)
    
    # Place left image fully on the left
    result[:, :w] = actual_left
    
    # Calculate where right image starts (with overlap)
    right_start = w - overlap_pixels
    
    # Simple cut - just place right image starting at right_start
    # This will overwrite the overlapping portion of the left image
    result[:, right_start:] = actual_right
    
    # Save result
    cv2.imwrite(output_path, result)


def main():
    folder_path = "../calib_images_USE"
    
    # Parse images by camera
    left_images, right_images = parse_image_files(folder_path)
    print(f"Found {len(left_images)} left camera images")
    print(f"Found {len(right_images)} right camera images\n")
    
    # Calibrate left camera
    left_calib = calibrate_single_camera(left_images, chessboard_size=(6, 8), square_size=27.0)
    
    if left_calib is None:
        print("Left camera calibration failed!")
        return
    
    # Calibrate right camera
    right_calib = calibrate_single_camera(right_images, chessboard_size=(6, 8), square_size=27.0)
    
    if right_calib is None:
        print("Right camera calibration failed!")
        return
    
    # Save calibration data
    np.savez('left_camera_calibration.npz',
             camera_matrix=left_calib['camera_matrix'],
             dist_coeffs=left_calib['dist_coeffs'],
             img_shape=left_calib['img_shape'],
             mean_error=left_calib['mean_error'])
    print("Saved left_camera_calibration.npz")
    
    np.savez('right_camera_calibration.npz',
             camera_matrix=right_calib['camera_matrix'],
             dist_coeffs=right_calib['dist_coeffs'],
             img_shape=right_calib['img_shape'],
             mean_error=right_calib['mean_error'])
    print("Saved right_camera_calibration.npz")
    
    # Create undistortion maps 
    left_map1, left_map2, left_new_mtx, left_roi = create_undistortion_maps(left_calib, alpha=0.0)
    right_map1, right_map2, right_new_mtx, right_roi = create_undistortion_maps(right_calib, alpha=0.0)
    
    print("\nLeft camera ROI:", left_roi)
    print("Right camera ROI:", right_roi)

    # Test undistortion on first image from each camera
    # Create output directory
    os.makedirs('undistorted_output', exist_ok=True)
    
    # Test left camera
    test_left = cv2.imread(left_images[0])
    undistorted_left = undistort_image(test_left, left_map1, left_map2)
    cv2.imwrite('undistorted_output/test_left_undistorted.jpg', undistorted_left)
    print(f"Saved undistorted_output/test_left_undistorted.jpg")
    
    # Test right camera
    test_right = cv2.imread(right_images[0])
    undistorted_right = undistort_image(test_right, right_map1, right_map2)
    cv2.imwrite('undistorted_output/test_right_undistorted.jpg', undistorted_right)
    print(f"Saved undistorted_output/test_right_undistorted.jpg")
    
    # Create side-by-side comparison
    comparison_left = np.hstack([test_left, undistorted_left])
    comparison_right = np.hstack([test_right, undistorted_right])
    cv2.imwrite('undistorted_output/comparison_left.jpg', comparison_left)
    cv2.imwrite('undistorted_output/comparison_right.jpg', comparison_right)
    print("Saved comparison images to undistorted_output folder.\n")
    
    # Compute homography from undistorted images
    homography = compute_homography_from_undistorted(
        left_images, right_images, 
        left_map1, left_map2, 
        right_map1, right_map2,
        num_pairs=10
    )
    
    if homography is None:
        print("Failed to compute homography!")
        return
    
    print("\nHomography matrix:")
    print(homography)
    
    # Save homography
    np.save('homography.npy', homography)
    print("Saved homography.npy")
    
    # Undistort and stitch all image pairs    
    os.makedirs('stitched_output', exist_ok=True)
    
    pairs = match_image_pairs(left_images, right_images)
    print(f"Found {len(pairs)} matching pairs to stitch")
    
    for i, (left_path, right_path) in enumerate(pairs):
        left_basename = os.path.basename(left_path)
        # Extract pair number from filename
        match = re.search(r'left_(\d+)_', left_basename)
        if match:
            pair_num = match.group(1)
            output_name = f"stitched_output/stitched_{pair_num}.jpg"
            
            stitch_undistorted_pair(
                left_path, right_path,
                left_map1, left_map2,
                right_map1, right_map2,
                homography,
                output_name
            )
    
    print(f"All {len(pairs)} pairs stitched!\n")

    print("\nCalibration Summary:")
    print(f"Left camera mean error: {left_calib['mean_error']:.4f} pixels")
    print(f"Right camera mean error: {right_calib['mean_error']:.4f} pixels\n")
    
    print(f"\nOutput:")
    print(f"  Calibration files: calibration_and_homography/fresh_calibration/")
    print(f"  Test images: calibration_and_homography/fresh_calibration/undistorted_output/")
    print(f"  Stitched images: calibration_and_homography/fresh_calibration/stitched_output/")


if __name__ == "__main__":
    main()
