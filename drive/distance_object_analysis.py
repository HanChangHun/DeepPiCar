import math
import sys
import json
import logging
import datetime
import argparse
from pathlib import Path
import time

import picar
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import cv2
import cv2.aruco as aruco

from pycoral.adapters import common
from pycoral.adapters import detect

from lane_navigation_model.lane_navigation_model_edgetpu import (
    LaneNavigationModelEdgeTPU,
)
from drive.utils import print_statistics, show_image
from drive.webcam_video_stream import WebcamVideoStream


from object_detection_model.objects_detection_model import ObjectDetectionModel


class DistanceEstimation:
    def __init__(self) -> None:

        with open("calibrationValues0.json") as f:
            cal_vals = json.load(f)
        self.cam_mtx = np.array(cal_vals["camera_matrix"])
        self.distor_factor = np.array(cal_vals["dist_coeff"])

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

    def process_tag(self, image):
        marker_length = 0.1
        corners, ids, _ = aruco.detectMarkers(image, self.aruco_dict)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, self.cam_mtx, self.distor_factor
        )

        if ids is not None:
            for i in range(0, ids.size):
                aruco.drawAxis(
                    image, self.cam_mtx, self.distor_factor, rvec[0], tvec[0], 0.06
                )
                cv2.putText(
                    image,
                    "%.1f cm -- %.0f degree"
                    % ((tvec[0][0][2] * 100), (rvec[0][0][2] / math.pi * 180)),
                    (0, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (244, 244, 244),
                )
                # print((int)(tvec[0][0][2] * 1000))

        return image


class ObjectDetectionVisualizer:
    def __init__(self) -> None:
        height = 180
        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms

    ############################
    # Frame processing steps
    ############################
    def detect_objects(self, frame):
        logging.debug("Detecting objects...")

        # call tpu for inference
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_RGB)
        _, scale = common.set_resized_input(
            self.interpreter,
            img_pil.size,
            lambda size: img_pil.resize(size, Image.ANTIALIAS),
        )
        self.interpreter.invoke()

        objects = detect.get_objects(
            self.interpreter, score_threshold=self.min_confidence, image_scale=scale
        )

        # scale_factor = self.width / img_pil.width
        # draw_objects(ImageDraw.Draw(img_pil), objects, scale_factor, self.labels)

        return objects, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


############################
# Utility Functions
############################


def draw_objects(draw, objs, scale_factor, labels):
    """Draws the bounding box and label for each object."""
    COLORS = np.random.randint(100, 255, size=(len(labels), 3), dtype=np.uint8)
    for obj in objs:
        bbox = obj.bbox
        color = tuple(int(c) for c in COLORS[obj.id])
        draw.rectangle(
            [
                (bbox.xmin * scale_factor, bbox.ymin * scale_factor),
                (bbox.xmax * scale_factor, bbox.ymax * scale_factor),
            ],
            outline=color,
            width=3,
        )
        font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
        draw.text(
            (bbox.xmin * scale_factor + 4, bbox.ymin * scale_factor + 4),
            "%s\n%.2f" % (labels.get(obj.id, obj.id), obj.score),
            fill=color,
            font=font,
        )


def main():
    pass


if __name__ == "__main__":
    main()
