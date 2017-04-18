import cv2
import numpy as np

from line import Left, Right

MAX_CURVATURE_RATIO = 2


class Lane():
    def __init__(self, left_line: Left, right_line: Right, image: np.ndarray = None):
        self.left_line = left_line
        self.right_line = right_line

    def sanity_check(self):
        curvature_ratio = self.left_line.get_curvature_radius() / self.right_line.get_curvature_radius()

        return (1 / MAX_CURVATURE_RATIO) <= curvature_ratio <= MAX_CURVATURE_RATIO

    def get_position(self):
        image_height_m = self.left_line.image_height_m
        left_fit_m = self.left_line.get_fit_m()
        left_position_m = left_fit_m[0] * image_height_m ** 2 + left_fit_m[1] * image_height_m + left_fit_m[2]
        right_fit_m = self.right_line.get_fit_m()
        right_position_m = right_fit_m[0] * image_height_m ** 2 + right_fit_m[1] * image_height_m + right_fit_m[2]
        position_m = (left_position_m + right_position_m) / 2
        center_m = self.left_line.image_width_m / 2

        return center_m - position_m

    def get_projection(self, image, color=(0, 255, 0)):
        projection = np.zeros_like(image).astype(np.uint8)

        points_left = self.left_line.get_trend_line()
        points_right = self.right_line.get_trend_line()
        points = np.array(np.vstack((points_left, np.flipud(points_right))), dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(projection, [points], color)

        return projection

    def get_curvature(self):
        return max(self.left_line.get_curvature_radius_m(), self.right_line.get_curvature_radius_m())
