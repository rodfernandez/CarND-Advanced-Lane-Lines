import glob

import cv2
import numpy as np
import peakutils

import util
from binary_threshold import combined_threshold
from lane import Lane
from line import Left, Right
from perspective_transform import warp
from undistort_image import undistort

MASK_WIDTH = int(1280 / 20)
MINIMUM_DISTANCE = 0
THRESHOLD = 0.0
WINDOW_COUNT = 9
X_METERS_PER_PIXEL = 3.67 / 820  # lane width ~ 3.67m, see: https://goo.gl/lzsRjT
Y_METERS_PER_PIXEL = 3.67 / 120  # lane marking lenght ~ 3.67m, see: https://goo.gl/D3OgRP


def annotate(image, lane):
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    lane.left_line.draw_line(annotated)
    lane.right_line.draw_line(annotated)

    lane.left_line.draw_points(annotated)
    lane.right_line.draw_points(annotated)

    return annotated


left_line = None
right_line = None


def apply_mask(image, mask_width=MASK_WIDTH):
    if left_line is None or right_line is None:
        return image

    mask = cv2.bitwise_or(left_line.get_mask(mask_width), right_line.get_mask(mask_width))
    return cv2.bitwise_and(image, (255), mask=mask)


def find_lane(image: np.ndarray, minimum_distance=MINIMUM_DISTANCE, smooth=False, threshold=THRESHOLD, use_mask=False):
    global left_line
    global right_line

    image_height, image_width = image.shape
    window_height = int(image_height / WINDOW_COUNT)
    image_center = int(image_width / 2)

    if use_mask:
        image = apply_mask(image)
    else:
        left_line = None
        right_line = None

    left_points, right_points = [], []

    for y1 in np.arange(image_height, 0, -window_height):
        y0 = y1 - window_height
        window = image[y0:y1, :]

        # find peaks using PeakUtils (http://pythonhosted.org/PeakUtils/)
        sum = np.sum(window, axis=0, dtype=np.float)
        if sum.max() > 0.0:
            sum -= peakutils.baseline(sum, deg=8)
        peak = peakutils.indexes(sum, thres=threshold, min_dist=minimum_distance)

        # sort left/right
        y = int((y0 + y1) / 2)
        for x in peak:
            if x <= image_center:
                left_points.append((y, x))
            else:
                right_points.append((y, x))

    left_line = Left(left_points, image_height, image_width, Y_METERS_PER_PIXEL, X_METERS_PER_PIXEL, smooth)
    right_line = Right(right_points, image_height, image_width, Y_METERS_PER_PIXEL, X_METERS_PER_PIXEL, smooth)

    lane = Lane(left_line, right_line, image)

    return lane


if __name__ == '__main__':
    test_images = glob.glob('../test_images/*.jpg')
    output_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)

        undistorted = undistort(image)
        warped = warp(undistorted, borderMode=cv2.BORDER_REFLECT)
        binary = combined_threshold(warped)

        lane = find_lane(binary, minimum_distance=640, threshold=0.1, use_mask=False)
        annotated = annotate(binary, lane=lane)

        output_images.append(warped)
        output_images.append(annotated)

    collage = util.collage(output_images, len(test_images), 2)
    cv2.imwrite('../output_images/lane_finder.png', collage)
