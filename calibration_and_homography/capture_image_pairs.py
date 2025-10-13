from picamera2 import Picamera2
import cv2
import time
import os

os.makedirs("calib_images", exist_ok=True)  # If the folder calib_images doesn’t exist, it will be created

cam0 = Picamera2(0)
cam1 = Picamera2(1)
# Start capturing frames internally so they’re ready when I ask for an image
cam0.start()
cam1.start()

for i in range(10):  # capture 10 pairs
    frame0 = cam0.capture_array()  # Returns a NumPy array of image data
    frame1 = cam1.capture_array()  # same thing
    cv2.imwrite(f"calib_images/left_{i:02d}.jpg", frame0)  
    cv2.imwrite(f"calib_images/right_{i:02d}.jpg", frame1)
    print(f"Captured pair {i}")
    time.sleep(2)

print("All pairs captured.")
