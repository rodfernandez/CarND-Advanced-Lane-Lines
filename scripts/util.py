import urllib.request

import cv2
import numpy as np


def convert_to_color(image):
    if image.dtype != np.uint8:
        image = denormalize(image)

    if image.shape[-1] == 3:
        return image
    else:
        return cv2.cvtColor(image, code=cv2.COLOR_GRAY2BGR)


def collage(src, rows, columns):
    src = np.array(list(map(lambda x: convert_to_color(x), src)))
    count, height, width, channels = src.shape

    dst = np.full((rows * height, columns * width, channels), 255, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            dst[i * height: (i + 1) * height, j * width: (j + 1) * width] = src[i * columns + j]

    return dst


def denormalize(img):
    return (img * 255).astype(dtype='uint8')


def gray_normalized(src):
    width, height, channels = src.shape
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(dst)
    dst = np.reshape(dst, (width, height, 1))
    return dst


def download_test_image():
    url = 'https://s3.amazonaws.com/udacity-sdc/advanced_lane_finding/signs_vehicles_xygrad.png'

    with urllib.request.urlopen(url) as f:
        image = cv2.imdecode(np.asarray(bytearray(f.read()), dtype='uint8'), cv2.IMREAD_COLOR)

    return image
