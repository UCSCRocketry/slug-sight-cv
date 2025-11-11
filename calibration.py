import os
import glob
import cv2
import numpy as np

def calibrate_camera(image_files, chessboard_size=(8,6), square_size=1.0):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size
    objpoints = []
    imgpoints = []
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                 cv2.CALIB_CB_FAST_CHECK)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def main():
    left_files  = sorted(glob.glob(os.path.join('calibration_and_homography', 'calib_images_USE', 'left*.jpg')))
    right_files = sorted(glob.glob(os.path.join('calibration_and_homography', 'calib_images_USE', 'right*.jpg')))
    
    if not left_files or not right_files:
        print("No calibration images found")
        return
    
    print(f"Left: {len(left_files)}, Right: {len(right_files)}")
    
    _, mtxL, distL, rvecsL, tvecsL = calibrate_camera(left_files)
    _, mtxR, distR, rvecsR, tvecsR = calibrate_camera(right_files)
    
    if len(left_files) == len(right_files):
        objp = np.zeros((8*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
        objp *= 1.0
        objpoints = []
        imgpointsL = []
        imgpointsR = []
        i = 0
        for lf, rf in zip(left_files, right_files):
            imgL = cv2.imread(lf)
            imgR = cv2.imread(rf)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            retL_tmp, cornersL = cv2.findChessboardCorners(grayL, (8,6), None)
            retR_tmp, cornersR = cv2.findChessboardCorners(grayR, (8,6), None)
            if retL_tmp and retR_tmp:
                i += 1
                objpoints.append(objp)
                cornersL2 = cv2.cornerSubPix(grayL, cornersL, (8,6), (-1,-1),
                                             criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                cornersR2 = cv2.cornerSubPix(grayR, cornersR, (8,6), (-1,-1),
                                             criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                imgpointsL.append(cornersL2)
                imgpointsR.append(cornersR2)
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate( 
            objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1], 
            criteria=(cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5),)
        print("Stereo calibration done")
    else:
        print("Skipping stereo calibration")
    
    np.savez('camera_calibration_results.npz',
             mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
             rvecsL=rvecsL, tvecsL=tvecsL, rvecsR=rvecsR, tvecsR=tvecsR)
    print("Results saved")

if __name__ == '__main__':
    main()
