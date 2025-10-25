import cv2
import numpy as np


chessboard_size = (6,7) #__> I don't remember the dimensions of the chessboard, so I'm using 6X7
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

all_corners_left = []
all_corners_right = []

for i in range(30):
    img1 = cv2.imread(f'calib_images/left_{i:02d}.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'calib_images/right_{i:02d}.jpg', cv2.IMREAD_GRAYSCALE)
    
    ret1, corners1 = cv2.findChessboardCorners(img1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(img2, chessboard_size, None)
    
    if ret1 and ret2:
        corners1 = cv2.cornerSubPix(img1, corners1, (11,11), (-1,-1), criteria)
        corners2 = cv2.cornerSubPix(img2, corners2, (11,11), (-1,-1), criteria)
        
        all_corners_left.append(corners1)
        all_corners_right.append(corners2)

src_points = np.vstack(all_corners_left).reshape(-1, 2)
dst_points = np.vstack(all_corners_right).reshape(-1, 2)

H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

np.save('homography_matrix.npy', H)