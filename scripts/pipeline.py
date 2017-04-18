import glob

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

import util
from binary_threshold import combined_threshold
from lane_finder import find_lane
from perspective_transform import unwarp, warp
from undistort_image import undistort

previous_lane = None


def process_image(image, minimum_distance=640, smooth=True, threshold=0.1, use_mask=True):
    global previous_lane
    undistorted = undistort(image)
    warped = warp(undistorted, borderMode=cv2.BORDER_REFLECT)
    binary = combined_threshold(warped)
    lane = find_lane(binary, minimum_distance=minimum_distance, smooth=smooth, threshold=threshold, use_mask=use_mask)
    if lane.sanity_check():
        previous_lane = lane
    elif previous_lane is not None:
        lane = previous_lane
    radius_label = "CURVATURE = {:.2e} m - POSITION= {:.2e} m".format(lane.get_curvature(), lane.get_position())
    cv2.putText(undistorted, radius_label, (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    unwarped_lane_projection = unwarp(lane.get_projection(image))
    return cv2.addWeighted(undistorted, 1, unwarped_lane_projection, 0.3, 0)


if __name__ == '__main__':
    test_images = glob.glob('../test_images/*.jpg')
    output_images = []

    for test_image in test_images:
        image = cv2.imread(test_image)
        result = process_image(image, smooth=False, threshold=0.0, use_mask=False)

        output_images.append(image)
        output_images.append(result)

    collage = util.collage(output_images, len(test_images), 2)
    cv2.imwrite('../output_images/pipeline.png', collage)

    input_clip = VideoFileClip('../project_video.mp4')
    output_clip = input_clip.fl_image(process_image)
    output_clip.write_videofile('../output.mp4', audio=False)
