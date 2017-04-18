import glob
import pickle

import cv2

import util

RESULT_PATH = '../data/camera_calibration.p'

camera_matrix = None
distortion_coefficients = None


def get_calibration():
    if camera_matrix is None or distortion_coefficients is None:
        with open(RESULT_PATH, 'rb') as file:
            return pickle.load(file)

    return camera_matrix, distortion_coefficients


def undistort(image):
    camera_matrix, distortion_coefficients = get_calibration()
    return cv2.undistort(image, camera_matrix, distortion_coefficients)


if __name__ == '__main__':
    test_images = glob.glob('../test_images/*.jpg')
    output_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        undistorted = undistort(image)

        output_images.append(image)
        output_images.append(undistorted)

    collage = util.collage(output_images, len(test_images), 2)
    cv2.imwrite('../output_images/undistorted.png', collage)
