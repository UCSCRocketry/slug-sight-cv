import cv2

# Try both camera indices (0 and 1)
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()
    if not ret0 or not ret1:
        break
    
    combined = cv2.hconcat([frame0, frame1])  # side-by-side
    cv2.imshow("Dual Camera Feed", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()
