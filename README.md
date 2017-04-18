# Advanced Lane Finding

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. The camera calibration images, test road images, and project videos are available in the project repository.

## Camera Calibration

> Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the [camera_calibration.py](scripts/camera_calibration.py) module, I wrote pipeline that loads the images from the [camera_cal](camera_cal) directory and applies `cv2.findChessboardCorners`, `cv2.drawChessboardCorners` to find the points required by `cv2.calibrateCamera` which in its turn calculates the camera matrix and distortion coefficients. These values are stored at [data/camera_calibration.p](/Users/rfernandez/git/rf/CarND-Advanced-Lane-Lines/data/camera_calibration.p), to avoid running calibration for the next steps.
 
The results can be seem bellow:

![Camera calibration](output_images/calibration.jpg)

## Pipeline (test images)

> Provide an example of a distortion-corrected image.

In the [undistort_image.py](/scripts/undistort_image.py) module, the calibration data is loaded and the `undistort` function is exposed. Here is the result of applying to the [test_images](test_images):

![Undistorted images](output_images/undistorted.png)

> Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

In the [binary_threshold.py](/scripts/binary_threshold.py) module, I reused and refactored the code from course material and exposed the following functions:

* [`color_threshold`](scripts/binary_threshold.py#L91): combines the thresholds for the HLS channels (25° < H < 93° - emphasis on yellow, L > 88% and S > 88%).

![Color threshold](output_images/color_threshold.png)

* [`absolute_threshold`](scripts/binary_threshold.py#L108): combines the `cv2.Sobel` thresholds for X (from 21% to 50%) and Y (from 25% to 50%) axis.

![Absolute threshold](output_images/absolute_threshold.png)

* [`gradient_threshold`](scripts/binary_threshold.py#L100): combines the `cv2.Sobel` thresholds for magnitude (from 32% to 57%) and directional (from 0° to 6°) gradients.

![Gradient threshold](output_images/gradient_threshold.png)

* [`combined_threshold`](scripts/binary_threshold.py#L116): combines the color, absolute and magnitude gradient thresholds (directional gradient threshold was excluded because I failed to find a range that would boost signal/noise).

![Combined threshold](output_images/combined_threshold.png)

> Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the [perspective_transform.py](/scripts/perspective_transform.py) module, I exposed the function [`warp`](scripts/perspective_transform.py#L27) which transforms the input image according to [coordinates](scripts/perspective_transform.py#L17) extracted from the [first test image](test_images/straight_lines1.jpg). It employs `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.

![Warped images](output_images/warped.png)

> Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the [lane_finder.py](/scripts/lane_finder.py) module, I expose the function [`find_lane`](scripts/lane_finder.py#L46) takes a binary warped image and execute the following steps:
 
  1. If there were previous lines found, applies a mask that will clear any pixel that is not with the [tolerance](scripts/lane_finder.py#L14) from the previous lines trend;
  2. Iterate through windows from the botton to the top of the image:
     1. Calculates the histogram of full pixels for in vertical axis for the current window, using `numpy.sum`;
     2. Finds the histogram peaks using the `indexes` function from [PeakUtils](http://pythonhosted.org/PeakUtils/) library;
     3. Sorts the peaks for left and right sides of the image;
     4. Maps the peaks indexes into the image's coordinate system.
  3. Creates instances of the [`Left`](scripts/line.py#L116) and [`Right`](scripts/line.py#L108) classes with the found coordinates;
     * The [`Line`](scripts/line.py#L7) base class exposes the `get_fit` method that should return a second degree polynomial fit; 
  4. Creates and returns an instance of the [`Lane`](scripts/lane.py#L9) class.

![Lane finder](output_images/lane_finder.png)

> Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

* The radius of curvature is calculated according the [formula provided in project instructions](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f) by the method [`calculate_curvature`](scripts/line.py#L41) from the `Line` class;
* The lane position is calculated by the method [`get_position`](scripts/lane.py#L19) from the `Lane` class;
* To calculate those values in meters, I had to estimate how to scale the image coordinates:
  * In the horizontal axis, I found the separation between lanes (3.67 meters according [this document]( https://goo.gl/lzsRjT)) measured about 820 pixels;
  * In the vertical axis, I found the lane markings length (3.67 meters according [this document](https://goo.gl/D3OgRP)) measured about 120 pixels;
  * These scaling parameters where defined as constants [X_METERS_PER_PIXEL](scripts/lane_finder.py#L18) and [Y_METERS_PER_PIXEL](lane_finder.py#L19).

> Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![Lane projection](output_images/pipeline.png)

## Pipeline (video)

> Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

Video output: [output.mp4](output.mp4)

## Discussion

> Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

* I spend a long time yak-shaving, since I wanted to make sure to understand and familiarize myself with all the components I leveraged in this pipeline:
  * Python;
  * Numpy;
  * OpenCV;
  * PeakUtils.
* I didn't read and leveraged any particular Python styling guidelines, but I refactored it a few times to improve readability and flexibility;
* A good chunk of the time was also spent in finding and fine tuning the parameters in the [binary_threshold.py](/scripts/binary_threshold.py) module;
* I also tried different algorithms for the sliding windows, including the [histogram](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/c41a4b6b-9e57-44e6-9df9-7e4e74a1a49a) and [convolution](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/a819d5fd-c00e-4e0e-b7d1-c1bee24b12a7) methods explained in course material, but at the end I got similar, if not better, results with a simple to read approach using the PeakUtils library;
* I am not happy with the curvature values, I would expect very high numbers (close to infinite) for straights, but the position values seem accurate enough in the centimeters order of magnitude;
* We can see in the output video that algorithm doesn't perform well in a few situations:
  * When there are surface transitions (from asphalt to concrete and vice-versa) and when there are shadows in the road:
    * In this situation, there is noise introduced by the vertical axis from the absolute threshold and the magnitude gradient threshold;
  * When there are bumps in the road:
    * I suspect this confuses the smoothing algorithm;
    * Image stabilization could prevent this issue;
* The pipeline also would have trouble if there where visible objects in the road or close to the lines;
* In general, I believe that a more advanced signal processing algorithm could make the pipeline more robust.
