## GStreamer Rocket Video Pipeline 🚀 🎥

### Overview

This GStreamer folder accounts for part of the transmission pipeline that sends stitched live video from two Raspberry Pi cameras onboard a rocket to a ground station over analog video (VTX/VRX). All image stitching and video processing happen onboard the Pi before transmission.

---

### The full live video transmission pipeline is:

```
Pi 5 (frame stitching)
   │
   ├──(HDMI)──> Digital-to-Analog Converter
   │
   ├──(RCA)──> VTX → VRX
   │
   ├──(RCA)──> Analog-to-Digital Converter
   │
   └──(HDMI)──> Ground Station Display
```

In this folder though, we build the GStreamer pipeline for sending stitched video out of the Pi 5 HDMI port...

### That sub-pipeline look like this:

_(This is what we will be working on in CV. Our embedded systems members will be handling most of the rest of the transmission process)_

1. Each camera sends **raw video frames** into GStreamer.
2. Each camera stream is sent to **Python stitching** process via appsink
   - Python stitching process:
     - Reads both frames
     - Applies homography and blending
     - Produces a stitched frame
3. Push the **stitched frame** back into **GStreamer**
4. GStreamer sends that stitched pair of frames to the Pi’s **HDMI port**

---

### Local testing:

Because we can only test this pipeline on the Pi 5 though, we will test a GStreamer pipeline simulation locally (on our laptops). Here is what that looks like:

```
One picture pair (doesn't matter which pair, but must be same picture number) from calib_images_USE folder
                        ⬇️
Stitch the two images (assumes existing homography matrix)
                        ⬇️
Send new stitched image from python process into a GStreamer sink
                        ⬇️
Display the stitched image in a window
```

---

### Requirements for using GStreamer in Python:

- GStreamer installed
- gst-launch-1.0 command-line utility to prototype pipelines
- gst-inspect-1.0 to list and inspect available plugins
