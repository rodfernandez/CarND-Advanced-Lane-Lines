import glob

import cv2
import numpy as np

import util
from undistort_image import undistort

WIDTH = 1280
HEIGHT = 720
BORDER = WIDTH / 6
SIZE = WIDTH, HEIGHT
DST = np.float32([
    [WIDTH - BORDER, 0], [WIDTH - BORDER, HEIGHT], [BORDER, HEIGHT], [BORDER, 0]
])
SRC = np.float32([
    [704, 461], [1036, 666], [274, 666], [577, 461]  # measured from undistorted test_images/straight_lines1.jpg
])
M = cv2.getPerspectiveTransform(SRC, DST)
M_INV = cv2.getPerspectiveTransform(DST, SRC)


def unwarp(image):
    return cv2.warpPerspective(image, M_INV, SIZE)


def warp(image, borderMode=cv2.BORDER_DEFAULT):
    return cv2.warpPerspective(image, M, SIZE, flags=cv2.INTER_LINEAR, borderMode=borderMode)


if __name__ == '__main__':
    test_images = glob.glob('../test_images/*.jpg')
    output_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        undistorted = undistort(image)
        warped = warp(undistorted)

        cv2.polylines(undistorted, [(SRC.astype(np.int64))], 1, (0, 0, 255), 3)
        cv2.polylines(warped, [(DST.astype(np.int64))], 1, (0, 0, 255), 3)

        output_images.append(undistorted)
        output_images.append(warped)

    collage = util.collage(output_images, len(test_images), 2)
    cv2.imwrite('../output_images/warped.png', collage)
