import gi
gi.require_version("Gst", "1.0") # need this version for gstreamer with opencv
from gi.repository import Gst, GLib
import numpy as np
import cv2
import time

Gst.init(None)

width, height, fps = 640, 480, 30 # define video properties

# camera capture pipeline with tee to split the stream
# one output goes to appsink for opencv
# last output goes to filesink to save raw video

# for pi, replace x264enc with v4l2h264enc
# also, replace avfvideosrc with libcamerasrc

cap_pipeline_str = (
    f"avfvideosrc device-index=0 ! "
    f"videoconvert ! video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
    "tee name=t "
    "t. ! queue ! appsink name=sink "
    "t. ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
    "mp4mux ! filesink location=output_raw.mp4"
)
#WILL NEED TO CHANGE FOR PI5 ^: saved location might need to be different, 

cap = Gst.parse_launch(cap_pipeline_str)
appsink = cap.get_by_name("sink")
cap.set_state(Gst.State.PLAYING)

# pipeline for saving after processing through opencv
save_pipeline_str = (
    f"appsrc name=src is-live=true block=true format=time "
    f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
    "videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
    "mp4mux ! filesink location=output_processed.mp4"
)
save_pipeline = Gst.parse_launch(save_pipeline_str)
appsrc = save_pipeline.get_by_name("src")
save_pipeline.set_state(Gst.State.PLAYING)

pts = 0  # running timestamp in nanoseconds, used for saving processed video
duration = Gst.SECOND // fps

try:
    while True:
        sample = appsink.emit("try_pull_sample", Gst.SECOND // fps)
        if not sample:
            continue

        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, np.uint8).reshape((h, w, 3))

        # sample opencv algorithm to be replaced with real use
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == 27:  # esc key to stop
            break

        # need to convert edges back to BGR for saving
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # moves out the processed frame so they can be saved
        buf_out = Gst.Buffer.new_allocate(None, edges_bgr.nbytes, None)
        buf_out.fill(0, edges_bgr.tobytes())
        buf_out.pts = pts
        buf_out.duration = duration
        pts += duration
        appsrc.emit("push-buffer", buf_out)

finally:
    # end of processed video save pipeline
    appsrc.emit("end-of-stream")
    bus_save = save_pipeline.get_bus()
    
    # wait for processed video end of stream
    while True:
        msg = bus_save.timed_pop(Gst.SECOND)
        if msg and msg.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            break
    
    # send end of stream to capture pipeline to finalize raw video
    cap.send_event(Gst.Event.new_eos())
    
    # checking for end of stream, error if not
    bus_cap = cap.get_bus()
    while True:
        msg = bus_cap.timed_pop(Gst.SECOND * 5)
        if msg and msg.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"Error: {err}, {debug}")
            break
    
    # stop both pipelines
    cap.set_state(Gst.State.NULL)
    save_pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
    
    # raw video saved as output_raw.mp4
    # processed video saved as output_processed.mp4
    # a bunch of issues with libraries for me on mac, but might be better for pi?
