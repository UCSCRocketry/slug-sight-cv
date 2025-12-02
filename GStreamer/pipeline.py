"""
Main Thread (GLib.MainLoop)          Capture Thread
┌──────────────────────────┐        ┌─────────────────────────┐
│                          │        │ while running:          │
│ Process pipeline events  │        │   frame_L = capture()   │
│ Handle buffers           │◄───────│   frame_R = capture()   │
│ Encode & save video      │ push   │   create buffers        │
│ Display on screen        │ data   │   set timestamps        │
│                          │        │   push both buffers ────┤
└──────────────────────────┘        └─────────────────────────┘
"""

# 1. Each camera module on the Pi sends a frame into GStreamer



# 2. GStreamer sends the two frames to my Python stitching process



# 3. GStreamer receives the new stitched image and:
# - Saves in MPEG on the Pi while streaming 
# - Sends the stitched image out to the Pi’s HDMI port


import sys
import threading
import time
from datetime import datetime

import gi
import numpy as np
from picamera2 import Picamera2

# TODO: Import stitching function from undistort-stitching.py

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

Gst.init(None)

class StitchingPipeline:
    def __init__(self, output_width=1280, output_height=480):
        """
        Stereo camera stitching pipeline for HDMI output.
        Combines threading (for dual cameras) with shutdown handling.
        """
        self.number_frames = 0
        self.fps = 30
        self.duration = int(1 / self.fps * Gst.SECOND)

        # Threading control variables
        self.running = False
        self.capture_thread = None

        # Output dimensions
        self.output_width = output_width
        self.output_height = output_height

        # Performance tracking lists
        self.stitch_times = []
        self.total_times = []

        # Initialize cameras
        self.left_cam = Picamera2(0)
        self.right_cam = Picamera2(1)

        # Configure left and right cameras
        config_left = self.left_cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)}
        )
        config_right = self.right_cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)}
        )
        
        self.left_cam.configure(config_left)
        self.right_cam.configure(config_right)
        
        self.left_cam.start()
        self.right_cam.start()
        print("Cameras started successfully.")

        pipeline_string = (
            f"appsrc name=stitched_src is-live=true block=true format=GST_FORMAT_TIME "
            f"caps=video/x-raw,format=RGB,width={self.output_width},height={self.output_height},framerate={self.fps}/1 "
            "! videoconvert "    # Convert between video formats if needed
            "! waylandsink fullscreen=true sync=false"    # Display video on Wayland compositor, shows frames as soon as possible
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get the appsrc element to push stitched frames into
        self.app_src = self.pipeline.get_by_name('stitched_src')

        if not self.stitched_src:
            print("Error: Could not find stitched_src element.")
            sys.exit(1)
        
        # For sanity and debugging
        print(f"Output resolution: {self.output_width}x{self.output_height}")

        def capture_stitch_push(self):
        """
        Runs in separate thread (captures from both cameras, stitches, and pushes to GStreamer).
        """
        while self.running:
            frame_start = time.perf_counter()
            
            try:
                # Step 1: Capture both frames (synchronized)
                capture_start = time.perf_counter()
                frame_left = self.left_cam.capture_array()  # TODO: SAVE AND PRINT TIME AT LEFT CAPTURE
                frame_right = self.right_cam.capture_array()  # TODO: SAVE AND PRINT TIME AT RIGHT CAPTURE
                capture_time = (time.perf_counter() - capture_start) * 1000  # time to capture both frames

                # Step 2: Stitch frames together
                stitch_start = time.perf_counter()
                # TODO: MAKE SURE stitch_frames IS IMPORTED and CORRECT in undistort-stitching.py
                stitched_frame = stitch_frames(frame_left, frame_right)
                stitch_time = (time.perf_counter() - stitch_start) * 1000  # time to stitch frames

                # Verify output dimensions
                if stitched_frame.shape[:2] != (self.output_height, self.output_width):
                    print(f"Warning: Stitched frame shape {stitched_frame.shape} "
                          f"doesn't match expected ({self.output_height}, {self.output_width}, 3)")

                # STEP 3: Create GStreamer buffer
                stitched_frame_data = stitched_frame.tobytes()

                # Make a buffer and fill with stitched frame data
                buffer = Gst.Buffer.new_allocate(None, len(stitched_frame_data), None)
                success, map_info = buffer.map(Gst.MapFlags.WRITE)

                if success:
                    map_info.data[:] = stitched_frame_data
                    buffer.unmap(map_info)
                else:
                    raise RuntimeError("Could not map buffer for writing")
                    map_info.data[:] = stitched_frame_data

                # Set buffer timestamps for proper playback timing
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)

                # Step 4: Push buffer into GStreamer pipeline
                return_val = self.app_src.emit('push-buffer', buffer)

                # Check return status code for errors in buffer pushing
                if retval != Gst.FlowReturn.OK:
                    print(f"Push buffer error: {retval}")
                    break
                
                # Onto the next frame
                self.number_frames += 1

                # Performance tracking
                total_time = (time.perf_counter() - frame_start) * 1000
                self.stitch_times.append(stitch_time)
                self.total_times.append(total_time)

                # Limit frame rate to self.fps (30 FPS)
                elapsed = (time.perf_counter() - frame_start)
                target_duration = 1.0 / self.fps
                if elapsed < target_duration:
                    time.sleep(target_duration - elapsed)

            except Exception as e:
                print(f"Error in capture_stitch_push: {e}")
                import traceback
                traceback.print_exc()
                break

        # Send EOS when done
        print("End-of-Stream (EOS) signal... The Stream Hath Ended.")
        self.stitched_src.emit('end-of-stream')

    def run(self):
        """Start the pipeline and begin processing."""
        
        print("Starting GStreamer pipeline...")
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Start capture/stitch thread (instead of callback)
        # When this thread starts, it runs the capture_stitch_push() function
        self.running = True
        self.capture_thread = threading.Thread(
            target=self.capture_stitch_push,
            daemon=True
        )
        self.capture_thread.start()

        # Start GLib Main Loop to handle GStreamer events
        self.loop = GLib.MainLoop()
        try:
            print("Pipeline running. Outputting to HDMI for VTX.")
            print("Press Ctrl+C to stop.")
            # Runs the main loop, code stays here until CTRL+C pressed
            self.loop.run()

        except KeyboardInterrupt:
            print("Keyboard interrupted by CTRL+C. Stopping pipeline...")
            
            # Stop capture thread
            self.running = False
            print("Waiting for capture thread to finish...")
            if self.capture_thread:
                self.capture_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
            
            # Stop GStreamer pipeline
            if self.loop and self.loop.is_running():
                self.loop.quit()

            print("Waiting for pipeline to finalize (processing EOS)...")

            # Waits 10 seconds for EOS or ERROR message, then times out
            bus = self.pipeline.get_bus()
            msg = bus.timed_pop_filtered(
                10 * Gst.SECOND,
                Gst.MessageType.EOS | Gst.MessageType.ERROR
            )

            # Handle EOS or ERROR message
            if msg:
                if msg.type == Gst.MessageType.EOS:
                    print("EOS received. Pipeline finalized successfully.")
                elif msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"GStreamer Error on shutdown: {err}, {debug}")
            else:
                print("Timed out waiting for EOS.")

            # Pring final performance statistics
            if self.stitch_times:
                print("\n" + "-" * 60)
                print("PERFORMANCE STATISTICS")
                print("-" * 60)
                print(f"Total frames processed: {self.number_frames}")
                print(f"Average stitch time: {np.mean(self.stitch_times):.2f}ms")
                print(f"Max stitch time: {np.max(self.stitch_times):.2f}ms")
                print(f"Min stitch time: {np.min(self.stitch_times):.2f}ms")
                print(f"Average total time: {np.mean(self.total_times):.2f}ms")
                print(f"Actual FPS: {1000.0/np.mean(self.total_times):.2f}")
                print("-" * 60)

        finally:
            print("Cleaning up resources...")
            self.pipeline.set_state(Gst.State.NULL)
            
            self.left_cam.stop()
            self.right_cam.stop()
            self.left_cam.close()
            self.right_cam.close()
            print("Cleanup complete.")


if __name__ == "__main__":
    # (May need to change dimensions to accomodate for our VTX)
    pipeline = StitchingPipeline(
        output_width=1280,  # Maybe adjust
        output_height=480   # Maybe adjust
    )

    pipeline.run()  # Runs the StitchingPipeline class
