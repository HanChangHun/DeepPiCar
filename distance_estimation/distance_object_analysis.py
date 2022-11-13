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

        with open("distance_estimation/calibrationValues0.json") as f:
            cal_vals = json.load(f)
        self.cam_mtx = np.array(cal_vals["camera_matrix"])
        self.distor_factor = np.array(cal_vals["dist_coeff"])

        # self.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        self.aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    def process_tag(self, image):
        marker_length = 0.1
        corners, ids, _ = aruco.detectMarkers(image, self.aruco_dict)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, self.cam_mtx, self.distor_factor
        )

        if ids is not None:
            for _ in range(0, ids.size):
                # aruco.drawAxis(
                #    image, self.cam_mtx, self.distor_factor, rvec[0], tvec[0], 0.06
                # )
                cv2.putText(
                    image,
                    "%.1f cm" % (tvec[0][0][2] * 100),
                    (10, self.height - 20),
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

        self.labels = read_label_file("object_detection_model/model/obj_det_labels.txt")

        self.min_confidence = 0.5
        self.num_of_objects = 3
        self.cur_obj = []

    def update_obj(self, objects):
        self.cur_obj = objects

    def detect_objects(self, frame):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_RGB)

        self.draw_objects(ImageDraw.Draw(img_pil), 1, self.labels)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_objects(self, draw, scale_factor, labels):
        """Draws the bounding box and label for each object."""
        COLORS = (0, 0, 255)
        font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
        for obj in self.cur_obj:
            bbox = obj[2]
            draw.rectangle(
                [
                    (bbox[0] * scale_factor, bbox[1] * scale_factor),
                    (bbox[2] * scale_factor, bbox[3] * scale_factor),
                ],
                outline=COLORS,
                width=3,
            )
            draw.text(
                (bbox[0] * scale_factor + 4, bbox[1] * scale_factor + 4),
                "%s\n%.2f" % (labels.get(obj[0], obj[0]), obj[1]),
                fill=COLORS,
                font=font,
            )


def create_video_recorder(path, fourcc, fps, cap):
    return cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (int(cap.get(3)), int(cap.get(4))),
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument("-v", "--video", type=str, help="e.g. /path/to/video/video.mp4")
    # parser.add_argument(
    #    "-d", "--data", type=str, help="e.g. /path/to/video/objects.json"
    # )
    parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()
    result_dir = Path(args.path)
    video_path = str(result_dir / "video.avi")
    data_path = result_dir / "objects.json"

    width = 854
    height = 480
    fps = 20.0

    cap = cv2.VideoCapture(video_path)

    with open(data_path) as f:
        object_data = json.load(f)

    obj_frames = [obj["frame_cnt"] for obj in object_data]
    objects = [obj["objects"] for obj in object_data]

    obj_det_model_path = (
        "experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite"
    )
    obj_det_vis = ObjectDetectionVisualizer(obj_det_model_path, width, height)
    dist_estimator = DistanceEstimator(width, height)

    res_save_path = video_path.replace(".avi", "_vis.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video_res = create_video_recorder(res_save_path, fourcc, fps, cap)

    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cnt in obj_frames:
                for obj in object_data:
                    if obj["frame_cnt"] == cnt:
                        obj_det_vis.update_obj(obj["objects"])

            # frame = dist_estimator.process_tag(frame)
            frame = obj_det_vis.detect_objects(frame)
            video_res.write(frame)
        else:
            break
        cnt += 1
    video_res.release()


if __name__ == "__main__":
    main()
