# Fresh Camera Calibration

This folder contains a clean implementation for calibrating two separate fisheye cameras independently.

## What This Does

1. **Parses image pairs** from `calib_images_USE` folder

   - Separates left camera images from right camera images
   - Each camera is calibrated independently

2. **Calibrates each camera separately**

   - Uses chessboard pattern detection (6x8 grid, 27mm squares)
   - Computes camera matrix and distortion coefficients for each camera
   - Both cameras have 120° horizontal FOV (fisheye lenses)

3. **Creates undistortion maps**

   - Generates remap matrices to correct fisheye distortion
   - Uses alpha=0.0 for maximum distortion removal (minimal black borders)

4. **Tests undistortion**
   - Applies undistortion to sample images from each camera
   - Saves before/after comparisons

## Results

**Left Camera:**

- 39 images successfully processed
- Mean reprojection error: 0.0976 pixels (excellent)

**Right Camera:**

- 39 images successfully processed
- Mean reprojection error: 0.0777 pixels (excellent)

## Output Files

- `left_camera_calibration.npz` - Left camera calibration parameters
- `right_camera_calibration.npz` - Right camera calibration parameters
- `undistorted_output/test_left_undistorted.jpg` - Sample undistorted left image
- `undistorted_output/test_right_undistorted.jpg` - Sample undistorted right image
- `undistorted_output/comparison_left.jpg` - Before/after comparison for left camera
- `undistorted_output/comparison_right.jpg` - Before/after comparison for right camera

## Usage

Run the calibration:

```bash
python separate_camera_calibration.py
```

## Next Steps

With the cameras now calibrated and undistortion maps created, you can:

1. Compute homography between the undistorted images for stitching
2. Create a real-time pipeline that undistorts and stitches frames
3. Integrate with GStreamer for video processing
