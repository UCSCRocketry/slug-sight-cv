# combined_stitch_and_stream.py

import glob
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import gi
import numpy as np

# --- GStreamer Setup ---
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

Gst.init(None)


def parse_image_files(folder_path):
    """Parse image files and organize them by camera (left/right)"""
    print(f"Parsing images in: {folder_path}")
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


def calibrate_single_camera(image_paths, camera_name, chessboard_size=(6, 8), square_size=27.0):
    """Calibrate a single camera using its images"""
    print(f"\n--- Calibrating {camera_name} Camera ---")
    
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
        if img_shape is None:
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
            # Quieter fail
            # print(f"(Note: No chessboard in {os.path.basename(img_path)})")
            pass
    
    if len(objpoints) < 10:
        print(f"ERROR: Only found {len(objpoints)} valid images for {camera_name}. Need at least 10.")
        return None
    
    print(f"Found {len(objpoints)} valid images for calibration.")

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print(f"{camera_name} calibration failed.")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    print(f"{camera_name} Mean Error: {mean_error:.4f} pixels")
    
    return {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'img_shape': img_shape,
        'mean_error': mean_error,
        'successful_images': successful_images
    }


def create_undistortion_maps(calib_data, alpha=0.0):
    """Create undistortion maps from calibration data"""
    mtx = calib_data['camera_matrix']
    dist = calib_data['dist_coeffs']
    img_shape = calib_data['img_shape']
    
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, alpha, img_shape)
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
    
    print("\n--- Computing Homography ---")
    pairs = match_image_pairs(left_images, right_images)
    pairs = pairs[:num_pairs]  # Use first N pairs
    
    all_src_pts = []
    all_dst_pts = []
        
    for left_path, right_path in pairs:
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        left_undist = undistort_image(left_img, left_map1, left_map2)
        right_undist = undistort_image(right_img, right_map1, right_map2)
        
        left_gray = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(right_gray, None)  # RIGHT is source
        kp2, des2 = orb.detectAndCompute(left_gray, None)   # LEFT is destination
        
        if des1 is None or des2 is None:
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:100]
        
        if len(good_matches) < 10:
            continue
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])  # RIGHT
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # LEFT
        
        all_src_pts.extend(src_pts)
        all_dst_pts.extend(dst_pts)
            
    if len(all_src_pts) < 20:
        print("ERROR: Not enough matches found to compute homography.")
        return None
    
    all_src_pts = np.array(all_src_pts)
    all_dst_pts = np.array(all_dst_pts)
    
    H, mask = cv2.findHomography(all_src_pts, all_dst_pts, cv2.RANSAC, 5.0)
    print(f"(Homography computed from {len(all_src_pts)} total feature matches)")
    
    return H


def stitch_undistorted_pair(left_path, right_path, left_map1, left_map2, 
                            right_map1, right_map2, homography):
    """
    Stitch a pair of undistorted images and RETURN the result
    Note: The 'left' and 'right' labels are swapped - left files are actually right camera
    """
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"Warning: Could not read {left_path} or {right_path}")
        return None

    # Undistort
    left_undist = undistort_image(left_img, left_map1, left_map2)
    right_undist = undistort_image(right_img, right_map1, right_map2)
    
    # SWAP: "left" files are actually the right camera view, and vice versa
    actual_left = right_undist   # right_undist is the actual left view
    actual_right = left_undist   # left_undist is the actual right view
    
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
    result[:, right_start:] = actual_right
    
    # Return the stitched image
    return result

class Stitcher:
    """
    This class handles all the calibration and stitching logic.
    It performs the heavy calibration on init, and then provides
    a simple method to get the next stitched frame.
    """
    def __init__(self, folder_path, chessboard_size=(6, 8), square_size=27.0):
        print("=== Initializing Stitcher ===")
        print("This may take a few moments...")
        
        # 1. Parse image files
        self.left_images, self.right_images = parse_image_files(folder_path)
        if not self.left_images or not self.right_images:
            raise FileNotFoundError(f"No image pairs found in {folder_path}")

        # 2. Calibrate cameras
        self.left_calib = calibrate_single_camera(self.left_images, "Left", chessboard_size, square_size)
        self.right_calib = calibrate_single_camera(self.right_images, "Right", chessboard_size, square_size)
        
        if self.left_calib is None or self.right_calib is None:
            raise RuntimeError("Camera calibration failed. Cannot proceed.")
            
        # 3. Create undistortion maps
        self.left_map1, self.left_map2, _, _ = create_undistortion_maps(self.left_calib, alpha=0.0)
        self.right_map1, self.right_map2, _, _ = create_undistortion_maps(self.right_calib, alpha=0.0)
        
        # 4. Compute homography
        self.homography = compute_homography_from_undistorted(
            self.left_images, self.right_images,
            self.left_map1, self.left_map2,
            self.right_map1, self.right_map2,
            num_pairs=10
        )
        if self.homography is None:
            raise RuntimeError("Homography computation failed. Cannot proceed.")
            
        # 5. Get list of all matching pairs to process
        self.pairs = match_image_pairs(self.left_images, self.right_images)
        self.frame_index = 0
        self.total_frames = len(self.pairs)
        
        print("\n=== Stitcher Initialized Successfully ===")
        print(f"Found {self.total_frames} stitched frames to produce.")

    def get_next_stitched_frame(self):
        """
        Stitches the next available image pair and returns the frame.
        Returns (True, frame) on success, (False, None) on end of stream.
        """
        if self.frame_index >= self.total_frames:
            # No more frames
            return False, None
            
        # Get paths for the current pair
        left_path, right_path = self.pairs[self.frame_index]
        
        # Stitch the frame
        stitched_frame = stitch_undistorted_pair(
            left_path, right_path,
            self.left_map1, self.left_map2,
            self.right_map1, self.right_map2,
            self.homography
        )
        
        self.frame_index += 1
        
        if stitched_frame is None:
            # Failed to stitch this pair, try the next one
            print(f"Warning: Failed to stitch frame {self.frame_index}.")
            return self.get_next_stitched_frame() # Recursive call to skip bad frame
            
        return True, stitched_frame


class video():
    def __init__(self):
        self.number_frames = 0
        self.fps = 10
        self.width = -1
        self.height = -1
        self.duration = 1 / self.fps * Gst.SECOND
        

        CALIBRATION_IMAGE_FOLDER = "../calib_images_USE"
        
        try:
            self.stitcher = Stitcher(folder_path=CALIBRATION_IMAGE_FOLDER)
        except Exception as e:
            print(f"Error initializing Stitcher: {e}")
            sys.exit(1)

        # Get the first frame to determine dimensions (width, height)
        ret, first_frame = self.stitcher.get_next_stitched_frame()
        if not ret:
            print("Error: Stitcher could not produce a single frame.")
            sys.exit(1)
            
        self.first_frame = first_frame # Save it to push later
        self.is_first_frame = True     # Flag to push the saved frame
        self.height, self.width, _ = first_frame.shape

        print(f"Stitched Source Initialized: {self.width}x{self.height} @ {self.fps} FPS")

        # --- GStreamer Pipeline (same as your script) ---
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.output_filename = f"stitched_recording_{timestamp}.ts"
        
        main_stream = f"appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
                      f" caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 " \
                      f"! videoconvert ! video/x-raw,format=I420 "

        tee_element = "! tee name=t "

        file_branch = "t. ! queue max-size-buffers=1 name=queue_file " \
                      "! x264enc tune=zerolatency bitrate=2000 speed-preset=fast " \
                      "! mpegtsmux " \
                      f"! filesink buffer-mode=2 location={self.output_filename} "


        display_branch = "t. ! queue name=queue_display " \
                         "! videoconvert " \
                         "! waylandsink sync=false "

        self.pipe = main_stream + tee_element + file_branch + display_branch
        
        print("\n--- GStreamer Pipeline ---")
        print(self.pipe)
        print("----------------------------")
        
        self.pipeline = Gst.parse_launch(self.pipe)
        self.loop = None
        appsrc = self.pipeline.get_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
        
        if not all(self.pipeline.get_by_name(n) for n in ['t', 'queue_file', 'queue_display']):
            print("Error: Could not find all necessary GStreamer elements...")
            sys.exit(1)


    def run(self):
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.set_state(Gst.State.PLAYING)
            
            self.loop = GLib.MainLoop()
            try:
                print(f"\nStreaming and saving STITCHED video to '{self.output_filename}'.")
                print("Press Ctrl+C to stop.")
                self.loop.run()
                
            except KeyboardInterrupt:
                print("\n Ctrl+C detected. Shutting down gracefully...")
    
                if self.loop and self.loop.is_running():
                    self.loop.quit()
    
                appsrc = self.pipeline.get_by_name('source')
                if appsrc:
                    print("Sending End-of-Stream (EOS) signal...")
                    appsrc.emit('end-of-stream')
                else:
                    print("Error: Could not find appsrc. Cannot send EOS gracefully.")
    
                print("Waiting for file to finalize (processing EOS)...")
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(
                    10 * Gst.SECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR
                )
                
                if msg:
                    if msg.type == Gst.MessageType.EOS:
                        print("EOS received. File finalized successfully.")
                    elif msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print(f"GStreamer Error on shutdown: {err}, {debug}")
                else:
                    print("Timed out waiting for EOS. File should still be usable.")
    
            finally:
                print("Cleaning up resources...")
                self.pipeline.set_state(Gst.State.NULL)
                # self.cap.release() # No longer needed
                print("Cleanup complete.")

    def on_need_data(self, src):

        # Check if we should push the pre-fetched first frame
        if self.is_first_frame:
            frame = self.first_frame
            ret = True
            self.is_first_frame = False # Clear the flag
        else:
            # Get the next frame from the stitcher
            ret, frame = self.stitcher.get_next_stitched_frame()
        
        if ret: # 'ret' is True if a frame was stitched successfully
            data = frame.tobytes() 
            
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            
            if self.number_frames % 10 == 0:
                print(f'Pushed stitched frame {self.number_frames}/{self.stitcher.total_frames}')
            
            if retval != Gst.FlowReturn.OK:
                print(retval)
            return True # Continue pushing data
        else:
            # End of stream (stitcher ran out of image pairs)
            src.emit('end-of-stream')
            print("End of stitcher stream detected.")
            return False # Stop the 'need-data' signal

if __name__ == "__main__":
    v = video()
    v.run()