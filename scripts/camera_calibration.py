import glob
import pickle

import cv2
import numpy as np

import util

GLOB_PATTERN = '../camera_cal/calibration*.jpg'
NX = 9
NY = 6
RESULT_PATH = '../data/camera_calibration.p'

if __name__ == "__main__":
    filenames = glob.glob(GLOB_PATTERN)

    source_images = []
    gray_images = []

    for filename in filenames:
        source_image = cv2.imread(filename)
        gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        source_images.append(source_image)
        gray_images.append(gray_image)

    mapped_images = []

    object_points = []
    image_points = []

    objp = np.zeros((NY * NX, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    for source_image, gray_image in zip(source_images, gray_images):
        ret, corners = cv2.findChessboardCorners(gray_image, (NX, NY), None)
        mapped_image = cv2.drawChessboardCorners(source_image.copy(), (NX, NY), corners, ret)
        mapped_images.append(mapped_image)

        if ret:
            object_points.append(objp)
            image_points.append(corners)

    image_size = gray_images[0].shape[::-1]

    ret, camera_matrix, distortion_coefficients, *rest = cv2.calibrateCamera(object_points, image_points, image_size,
                                                                             None, None)

    with open(RESULT_PATH, 'wb') as f:
        pickle.dump((camera_matrix, distortion_coefficients), f)

    undistorted_images = []

    for source_image in source_images:
        undistorted_image = cv2.undistort(source_image.copy(), camera_matrix, distortion_coefficients)
        undistorted_images.append(undistorted_image)

    calibration_images = []

    for source_image, mapped_image, undistorted_image in zip(source_images, mapped_images, undistorted_images):
        calibration_images.append(source_image)
        calibration_images.append(mapped_image)
        calibration_images.append(undistorted_image)

    collage = util.collage(calibration_images, len(source_images), 3)
    cv2.imwrite('../output_images/calibration.jpg', collage)
