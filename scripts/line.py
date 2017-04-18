import cv2
import numpy as np

PREVIOUS_FIT_COUNT = 17


class Line:
    no_fit = np.array([0, 0, 0], dtype=np.float)
    previous_fit = []

    def __init__(self, points: np.ndarray, image_height: int, image_width: int, y_meters_per_pixel: int,
                 x_meters_per_pixel: int, smooth=False):
        self.points = np.array(points)
        self.image_height = image_height
        self.image_width = image_width
        self.y_meters_per_pixel = y_meters_per_pixel
        self.x_meters_per_pixel = x_meters_per_pixel

        if len(self.points) >= 2:
            self.fit = np.polyfit(self.points[:, 0], self.points[:, 1], 2)
        else:
            self.fit = self.no_fit

        if smooth:
            if len(self.previous_fit) >= PREVIOUS_FIT_COUNT:
                self.previous_fit.pop(0)

            self.previous_fit.append(self.fit)

        self.image_height_m = image_height * y_meters_per_pixel
        self.image_width_m = image_width * x_meters_per_pixel
        self.scale = np.array([y_meters_per_pixel, x_meters_per_pixel])
        self.points_m = self.points * self.scale

        if len(self.points) >= 2:
            self.fit_m = np.polyfit(self.points_m[:, 0], self.points_m[:, 1], 2)
        else:
            self.fit_m = self.no_fit * x_meters_per_pixel

    @staticmethod
    def calculate_curvature(fit, image_height):
        A, B = fit[0], fit[1]
        y = image_height

        return ((1 + (2 * A * y + B) ** 2) ** 1.5) / np.absolute(2 * A)

    def draw_points(self, image, radius=10, color=(0, 0, 255)):
        for cy, cx in self.points:
            cv2.circle(image, (int(cx), int(cy)), radius, color, -1)

        return image

    def draw_line(self, image, color=(0, 255, 255), thickness=5):
        pts = self.get_trend_line()
        cv2.polylines(image, [pts], False, color, thickness)
        return image

    def draw_windows(self, image, window_width, window_height, color=(0, 255, 0)):
        for cy, cx in self.points:
            x0 = int(cx - window_width / 2)
            y0 = int(cy - window_height / 2)
            x1 = int(x0 + window_width)
            y1 = int(y0 + window_height)

            cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)

        return image

    def get_curvature_radius(self):
        return Line.calculate_curvature(self.get_fit(), self.image_height)

    def get_curvature_radius_m(self):
        return Line.calculate_curvature(self.get_fit_m(), self.image_height_m)

    def is_detected(self):
        if len(self.previous_fit) >= PREVIOUS_FIT_COUNT:
            mean = np.mean(self.previous_fit, axis=0)
            std = np.std(self.previous_fit, axis=0)
            tolerance = 0.5 * std
            greater = np.greater(self.fit, mean - tolerance)
            less = np.less(self.fit, mean + tolerance)

            return np.all(greater) and np.all(less)

        return True

    def get_fit(self):
        if self.is_detected():
            return self.fit
        else:
            return np.mean(self.previous_fit, axis=0)

    def get_fit_m(self):
        return self.fit_m

    def get_mask(self, thickness=100):
        mask = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.draw_line(mask, (255, 255, 255), thickness)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    def get_trend_line(self):
        fit = self.get_fit()
        y = np.linspace(0, self.image_height - 1, self.image_height, dtype=np.int32)
        x = np.array(fit[0] * y ** 2 + fit[1] * y + fit[2], dtype=np.int32)
        return np.array(np.transpose(np.vstack([x, y])), dtype=np.int32)


class Right(Line):
    no_fit = np.array([0, 0, 1280 * 5 / 6], dtype=np.float)
    previous_fit = []

    def __init__(self, *args, **kwargs):
        super(Right, self).__init__(*args, **kwargs)


class Left(Line):
    no_fit = np.array([0, 0, 1280 / 6], dtype=np.float)
    previous_fit = []

    def __init__(self, *args, **kwargs):
        super(Left, self).__init__(*args, **kwargs)
