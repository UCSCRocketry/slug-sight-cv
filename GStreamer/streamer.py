import os
import signal
import sys
import threading
from datetime import datetime

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
        
        # --- PiCamera2 Setup ---
        self.picam2 = Picamera2()
        self.config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)}
        )
        self.picam2.configure(self.config)
        self.picam2.start()

        # Timestamping (So every file has a different name)
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.output_filename = f"recording_{timestamp}.ts"
        
        # --- GStreamer Pipeline with Tee and File Sink ---
        main_stream = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
                      " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
                      "! videoconvert ! video/x-raw,format=I420 ".format(self.fps)

        tee_element = "! tee name=t "

        # 3. File Branch (Saves as H.264 in MPEG-TS container)
        file_branch = "t. ! queue max-size-buffers=1 name=queue_file " \
                      "! x264enc tune=zerolatency bitrate=2000 speed-preset=fast " \
                      "! mpegtsmux " \
                      f"! filesink buffer-mode=2 location={self.output_filename} " # <-- Used f-string

        # 4. Display Branch (Local Window)
        network_branch = "t. ! queue name=queue_display " \
                         "! videoconvert " \
                         "! waylandsink fullscreen=true sync=false "

        self.pipe = main_stream + tee_element + file_branch + network_branch
        
        # --- GStreamer Initialization ---
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
                # <-- 4. UPDATE PRINT MESSAGE ---
                print(f"Streaming and saving video to '{self.output_filename}'. Press Ctrl+C to stop.")
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
                self.picam2.stop()
                self.picam2.close()
                print("Cleanup complete.")


    def on_need_data(self, src, lenght):
        frame = self.picam2.capture_array() 
        
        if frame is not None:
            data = frame.tobytes() 
            
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            
            if self.number_frames % 50 == 0:
                print('Pushed buffer, frame {}'.format(self.number_frames))
            
            if retval != Gst.FlowReturn.OK:
                print(retval)
            return True
        else:
            src.emit('end-of-stream')
            print("End of camera stream detected.")
            return False

if __name__ == "__main__":
    v = video()
    v.run()