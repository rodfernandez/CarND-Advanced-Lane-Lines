import glob

import cv2
import numpy as np

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
import util
from perspective_transform import warp
from undistort_image import undistort


def _absolute_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    dx = 1 if orient == 'x'else 0
    dy = 1 if orient == 'y'else 0
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Return the result
    return (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])


def absolute_x_threshold(image, sobel_kernel, threshold):
    return _absolute_threshold(image, orient='x', sobel_kernel=sobel_kernel, thresh=threshold)


def absolute_y_threshold(image, sobel_kernel, threshold):
    return _absolute_threshold(image, orient='y', sobel_kernel=sobel_kernel, thresh=threshold)


# Define a function to threshold an image for a given range and Sobel kernel
def gradient_direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1.0

    # Return the binary image
    return binary_output.astype(dtype=float)


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def gradient_magnitude_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag > thresh[0]) & (gradmag < thresh[1])] = 1

    # Return the binary image
    return binary_output.astype(dtype=float)


def h_threshold(image, threshold=(0, 255)):
    h = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HLS)[:, :, 0]
    return cv2.inRange(h, threshold[0], threshold[1]) / 255


def l_threshold(image, threshold=(0, 255)):
    l = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HLS)[:, :, 1]
    return cv2.inRange(l, threshold[0], threshold[1]) / 255


def s_threshold(image, threshold=(0, 255)):
    s = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HLS)[:, :, 2]
    return cv2.inRange(s, threshold[0], threshold[1]) / 255


def color_threshold(image):
    h_thresholded = h_threshold(image, (18, 66))
    l_thresholded = l_threshold(image, (223, 255))
    s_thresholded = s_threshold(image, (223, 255))
    color_thresholded = (h_thresholded == 1) | (l_thresholded == 1) | (s_thresholded == 1)

    return color_thresholded, h_thresholded, l_thresholded, s_thresholded


def gradient_threshold(image):
    g_thresholded = gradient_magnitude_threshold(image, 31, (82, 146))
    d_thresholded = gradient_direction_threshold(image, 31, (0.0, np.pi / 30))
    gd_thresholded = (g_thresholded == 1) | (d_thresholded == 1)

    return gd_thresholded, g_thresholded, d_thresholded


def absolute_threshold_(image):
    x_thresholded = absolute_x_threshold(image, 31, (47, 127))
    y_thresholded = absolute_y_threshold(image, 31, (56, 127))
    xy_thresholded = (x_thresholded == 1) | (y_thresholded == 1)

    return xy_thresholded, x_thresholded, y_thresholded


def _combined_threshold(image):
    color_thresholded, h_thresholded, l_thresholded, s_thresholded = color_threshold(image)
    xy_thresholded, x_thresholded, y_thresholded = absolute_threshold_(image)
    gd_thresholded, g_thresholded, d_thresholded = gradient_threshold(image)

    combined = (color_thresholded == 1) | (xy_thresholded == 1) | (g_thresholded == 1)
    combined = util.denormalize(combined)

    return combined, color_thresholded, xy_thresholded, g_thresholded


def combined_threshold(image):
    return _combined_threshold(image)[0]


if __name__ == '__main__':
    test_images = glob.glob('../test_images/*.jpg')

    color_threshold_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        image = warp(undistort(image))
        color_thresholded, h_thresholded, l_thresholded, s_thresholded = color_threshold(image)

        color_threshold_images.append(image)
        color_threshold_images.append(h_thresholded)
        color_threshold_images.append(l_thresholded)
        color_threshold_images.append(s_thresholded)
        color_threshold_images.append(color_thresholded)

    collage = util.collage(color_threshold_images, len(test_images), 5)
    cv2.imwrite('../output_images/color_threshold.png', collage)

    absolute_threshold_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        image = warp(undistort(image))
        xy_thresholded, x_thresholded, y_thresholded = absolute_threshold_(image)

        absolute_threshold_images.append(image)
        absolute_threshold_images.append(x_thresholded)
        absolute_threshold_images.append(y_thresholded)
        absolute_threshold_images.append(xy_thresholded)

    collage = util.collage(absolute_threshold_images, len(test_images), 4)
    cv2.imwrite('../output_images/absolute_threshold.png', collage)

    gradient_threshold_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        image = warp(undistort(image))
        gd_thresholded, g_thresholded, d_thresholded = gradient_threshold(image)

        gradient_threshold_images.append(image)
        gradient_threshold_images.append(g_thresholded)
        gradient_threshold_images.append(d_thresholded)
        gradient_threshold_images.append(gd_thresholded)

    collage = util.collage(gradient_threshold_images, len(test_images), 4)
    cv2.imwrite('../output_images/gradient_threshold.png', collage)

    combined_threshold_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        image = warp(undistort(image))
        combined, color_thresholded, xy_thresholded, gd_thresholded = _combined_threshold(image)

        combined_threshold_images.append(image)
        combined_threshold_images.append(color_thresholded)
        combined_threshold_images.append(xy_thresholded)
        combined_threshold_images.append(gd_thresholded)
        combined_threshold_images.append(combined)

    collage = util.collage(combined_threshold_images, len(test_images), 5)
    cv2.imwrite('../output_images/combined_threshold.png', collage)
