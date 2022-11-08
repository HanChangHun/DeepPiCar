import math
import json
import logging
import argparse
from pathlib import Path
import datetime
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import cv2
import cv2.aruco as aruco

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file


class DistanceEstimator:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

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
            for _ in range(0, ids.size):
                aruco.drawAxis(
                    image, self.cam_mtx, self.distor_factor, rvec[0], tvec[0], 0.06
                )
                cv2.putText(
                    image,
                    "%.1f cm" % (tvec[0][0][2] * 100),
                    (10, self.height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        return image


class ObjectDetectionVisualizer:
    def __init__(self, model_path, width, height) -> None:
        self.width = width
        self.height = height

        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.labels = read_label_file("object_detection_model/model/obj_det_labels.txt")

        self.min_confidence = 0.5
        self.num_of_objects = 3

    def detect_objects(self, frame):
        logging.debug("Detecting objects...")

        # call tpu for inference
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_RGB)
        _, scale = common.set_resized_input(
            self.interpreter,
            img_pil.size,
            lambda size: img_pil.resize(size, Image.Resampling.LANCZOS),
        )
        self.interpreter.invoke()

        objects = detect.get_objects(
            self.interpreter, score_threshold=self.min_confidence, image_scale=scale
        )

        scale_factor = self.width / img_pil.width
        draw_objects(ImageDraw.Draw(img_pil), objects, scale_factor, self.labels)

        return objects, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


############################
# Utility Functions
############################
def draw_objects(draw, objs, scale_factor, labels):
    """Draws the bounding box and label for each object."""
    COLORS = (0, 0, 255)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle(
            [
                (bbox.xmin * scale_factor, bbox.ymin * scale_factor),
                (bbox.xmax * scale_factor, bbox.ymax * scale_factor),
            ],
            outline=COLORS,
            width=3,
        )
        draw.text(
            (bbox.xmin * scale_factor + 4, bbox.ymin * scale_factor + 4),
            "%s\n%.2f" % (labels.get(obj.id, obj.id), obj.score),
            fill=COLORS,
            font=font,
        )


def create_video_recorder(path, fourcc, cap):
    return cv2.VideoWriter(
        path,
        fourcc,
        5.0,
        (int(cap.get(3)), int(cap.get(4))),
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--filepath", type=str, help="e.g. /path/to/video/video.mp4"
    )
    args = parser.parse_args()
    filepath = args.filepath

    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_FPS, 5)

    obj_det_model_path = (
        "experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite"
    )
    obj_det_vis = ObjectDetectionVisualizer(obj_det_model_path, 640, 360)
    dist_estimator = DistanceEstimator(640, 360)

    res_save_path = filepath.replace(".avi", "_vis.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video_res = create_video_recorder(res_save_path, fourcc, cap)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            _, frame = obj_det_vis.detect_objects(frame)
            frame = dist_estimator.process_tag(frame)
            video_res.write(frame)
        else:
            break


if __name__ == "__main__":
    main()
