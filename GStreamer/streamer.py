import os
import signal
import sys
import threading
from datetime import datetime
import numpy as np

import cv2
import gi
from picamera2 import Picamera2

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

Gst.init(None)

class video():
    def __init__(self):
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND
        
        # --- Load Calibration Data and Homography ---
        calib_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'calibration_and_homography', 'undistort-and-stitch')
        
        # Check if calibration directory exists
        if not os.path.exists(calib_dir):
            print(f"Error: Calibration directory not found: {calib_dir}")
            print("Please run calibration first (calibration_and_homography/undistort-and-stitch/undistorting-stitching.py)")
            sys.exit(1)
        
        # Load left camera calibration
        left_calib_path = os.path.join(calib_dir, 'left_camera_calibration.npz')
        if not os.path.exists(left_calib_path):
            print(f"Error: Left camera calibration file not found: {left_calib_path}")
            sys.exit(1)
        left_calib = np.load(left_calib_path)
        left_mtx = left_calib['camera_matrix']
        left_dist = left_calib['dist_coeffs']
        left_img_shape = tuple(left_calib['img_shape'])
        
        # Load right camera calibration
        right_calib_path = os.path.join(calib_dir, 'right_camera_calibration.npz')
        if not os.path.exists(right_calib_path):
            print(f"Error: Right camera calibration file not found: {right_calib_path}")
            sys.exit(1)
        right_calib = np.load(right_calib_path)
        right_mtx = right_calib['camera_matrix']
        right_dist = right_calib['dist_coeffs']
        right_img_shape = tuple(right_calib['img_shape'])
        
        # Create undistortion maps
        alpha = 0.0  # Minimum unwanted pixels
        self.left_map1, self.left_map2, _, _ = cv2.initUndistortRectifyMap(
            left_mtx, left_dist, None,
            cv2.getOptimalNewCameraMatrix(left_mtx, left_dist, left_img_shape, alpha, left_img_shape)[0],
            left_img_shape, cv2.CV_16SC2
        )
        self.right_map1, self.right_map2, _, _ = cv2.initUndistortRectifyMap(
            right_mtx, right_dist, None,
            cv2.getOptimalNewCameraMatrix(right_mtx, right_dist, right_img_shape, alpha, right_img_shape)[0],
            right_img_shape, cv2.CV_16SC2
        )
        
        # Load homography matrix (currently not used in simple cut stitching, 
        # but available for future homography-based warping if needed)
        homography_path = os.path.join(calib_dir, 'homography.npy')
        if not os.path.exists(homography_path):
            print(f"Error: Homography file not found: {homography_path}")
            sys.exit(1)
        self.homography = np.load(homography_path)
        
        print("Calibration data and homography loaded successfully")
        
        # --- PiCamera2 Setup for TWO cameras ---
        self.picam2_left = Picamera2(0)  # Left camera (camera index 0)
        self.picam2_right = Picamera2(1)  # Right camera (camera index 1)
        
        # Configure both cameras
        config_left = self.picam2_left.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)}
        )
        config_right = self.picam2_right.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)}
        )
        
        self.picam2_left.configure(config_left)
        self.picam2_right.configure(config_right)
        
        self.picam2_left.start()
        self.picam2_right.start()
        
        print("Both cameras initialized and started")

        # Timestamping (So every file has a different name)
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.output_filename = f"recording_{timestamp}.ts"
        
        # Stitched frame dimensions: 2*width - overlap (15% overlap)
        # 640 * 2 - int(640 * 0.15) = 1280 - 96 = 1184 pixels
        stitched_width = 640 * 2 - int(640 * 0.15)  # 1184 pixels
        stitched_height = 480
        
        # --- GStreamer Pipeline with Tee for HDMI output and file recording ---
        # appsrc receives stitched frames from Python
        main_stream = f"appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
                      f"caps=video/x-raw,format=BGR,width={stitched_width},height={stitched_height},framerate={self.fps}/1 " \
                      f"! videoconvert ! video/x-raw,format=I420 "

        tee_element = "! tee name=t "

        # File Branch (Saves as H.264 in MPEG-TS container)
        file_branch = "t. ! queue max-size-buffers=1 name=queue_file " \
                      "! x264enc tune=zerolatency bitrate=2000 speed-preset=fast " \
                      "! mpegtsmux " \
                      f"! filesink buffer-mode=2 location={self.output_filename} "

        # HDMI Branch (for VTX/VRX system - outputs to HDMI port)
        hdmi_branch = "t. ! queue name=queue_hdmi " \
                      "! videoconvert " \
                      "! waylandsink fullscreen=true sync=false "

        self.pipe = main_stream + tee_element + file_branch + hdmi_branch
        
        # --- GStreamer Initialization ---
        self.pipeline = Gst.parse_launch(self.pipe)
        self.loop = None
        appsrc = self.pipeline.get_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
        
        if not all(self.pipeline.get_by_name(n) for n in ['t', 'queue_file', 'queue_hdmi']):
            print("Error: Could not find all necessary GStreamer elements...")
            sys.exit(1)
        
        # Thread synchronization for frame capture
        self.lock = threading.Lock()
        self.left_frame = None
        self.right_frame = None
        self.frame_ready = threading.Event()
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()


    def capture_frames(self):
        """Continuously capture frames from both cameras in a separate thread"""
        while True:
            try:
                # Capture frames from both cameras
                left_frame = self.picam2_left.capture_array()
                right_frame = self.picam2_right.capture_array()
                
                # Convert RGB888 to BGR for OpenCV processing
                left_frame_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
                right_frame_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
                
                # Update frames with lock
                with self.lock:
                    self.left_frame = left_frame_bgr
                    self.right_frame = right_frame_bgr
                    self.frame_ready.set()
                    
            except Exception as e:
                print(f"Error capturing frames: {e}")
                break
    
    def undistort_image(self, img, map1, map2):
        """Apply undistortion to an image"""
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    
    def stitch_frames(self, left_img, right_img):
        """Stitch two undistorted frames together
        
        Note: Based on the calibration code, 'left' files are actually right camera view
        and vice versa. So we swap them here.
        """
        # Undistort both frames
        left_undist = self.undistort_image(left_img, self.left_map1, self.left_map2)
        right_undist = self.undistort_image(right_img, self.right_map1, self.right_map2)
        
        # SWAP: "left" files are actually the right camera view, and vice versa
        # Based on the calibration code comments
        actual_left = right_undist  # right_undist is the actual left view
        actual_right = left_undist  # left_undist is the actual right view
        
        h, w = actual_left.shape[:2]
        
        # Calculate overlap (15% as per calibration code)
        overlap_pixels = int(w * 0.15)
        
        # Create canvas that accounts for overlap
        canvas_width = w * 2 - overlap_pixels
        result = np.zeros((h, canvas_width, 3), dtype=np.uint8)
        
        # Place left image fully on the left
        result[:, :w] = actual_left
        
        # Calculate where right image starts (with overlap)
        right_start = w - overlap_pixels
        
        # Simple cut - place right image starting at right_start
        # This overwrites the overlapping portion of the left image
        result[:, right_start:] = actual_right
        
        return result

    def run(self):
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.loop = GLib.MainLoop()
        try:
            print(f"Streaming stitched video to HDMI and saving to '{self.output_filename}'. Press Ctrl+C to stop.")
            self.loop.run()
            
        except KeyboardInterrupt:
            print("\n Ctrl+C detected. Shutting down gracefully...")
    
            if self.loop.is_running():
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
            self.picam2_left.stop()
            self.picam2_left.close()
            self.picam2_right.stop()
            self.picam2_right.close()
            print("Cleanup complete.")


    def on_need_data(self, src, lenght):
        """Called when GStreamer needs more data - stitches frames and pushes them"""
        # Wait for frames to be ready
        if not self.frame_ready.wait(timeout=1.0):
            print("Warning: Timeout waiting for frames")
            return False
        
        # Get frames with lock
        with self.lock:
            if self.left_frame is None or self.right_frame is None:
                self.frame_ready.clear()
                return False
            
            left_frame = self.left_frame.copy()
            right_frame = self.right_frame.copy()
            self.frame_ready.clear()  # Clear event so we wait for next frame pair
        
        # Stitch the frames together
        try:
            stitched_frame = self.stitch_frames(left_frame, right_frame)
        except Exception as e:
            print(f"Error stitching frames: {e}")
            return False
        
        # Convert stitched frame to bytes
        data = stitched_frame.tobytes()
        
        # Create GStreamer buffer
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        self.number_frames += 1
        
        # Push buffer to pipeline
        retval = src.emit('push-buffer', buf)
        
        if self.number_frames % 50 == 0:
            print(f'Pushed stitched buffer, frame {self.number_frames}')
        
        if retval != Gst.FlowReturn.OK:
            print(f"Warning: Push buffer returned {retval}")
            if retval == Gst.FlowReturn.FLUSHING:
                return False
        
        return True

if __name__ == "__main__":
    v = video()
    v.run()