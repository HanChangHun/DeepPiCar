import math
import sys
import json
import logging
import datetime
import argparse
from pathlib import Path
import time

import picar

import numpy as np
import cv2
import cv2.aruco as aruco

from lane_navigation_model.lane_navigation_model_edgetpu import (
    LaneNavigationModelEdgeTPU,
)
from drive.utils import print_statistics, show_image
from drive.webcam_video_stream import WebcamVideoStream
from drive.video_recoder import VideoRecoder


from object_detection_model.objects_detection_model import ObjectDetectionModel


class DeepPiCar:
    def __init__(
        self, initial_speed=0, show_image=False, video_save_dir=Path("drive/data")
    ):
        """Init camera and wheels"""
        logging.info("Creating a DeepPiCar...")
        self.show_image = show_image

        picar.setup()
        self.initial_speed = initial_speed

        # logging.debug("Set up camera")
        self.screen_width = 854
        self.screen_height = 480
        self.fps = 10.0

        self.camera = WebcamVideoStream(
            -1, self.screen_width, self.screen_height, self.fps
        ).start()

        logging.debug("Set up back wheels")
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        logging.debug("Set up front wheels")
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0  # calibrate servo to center
        self.front_wheels.turn(
            90
        )  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        self.obj_det_model_path = (
            "experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite"
        )
        logging.debug("Set up object detection model")
        self.obj_det_model = ObjectDetectionModel(
            self,
            model_path=self.obj_det_model_path,
            speed_limit=self.initial_speed,
            width=self.screen_width,
            height=self.screen_height,
        )
        self.warmup_obj_det()

        logging.debug("Set up video stream")
        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.out_dir = Path(video_save_dir / f"{date_str}")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.video_orig = VideoRecoder(
            self.camera, self.out_dir / "video.avi", self.fps
        ).start()

        # self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        # self.video_orig = self.create_video_recorder(
        #     str(self.video_save_dir / "car_video.avi")
        # )
        # self.video_lane = self.create_video_recorder(
        #     str(self.video_save_dir / "car_video_lane.avi")
        # )
        # self.video_objs = self.create_video_recorder(
        #     str(self.video_save_dir / "car_video_objs.avi")
        # )
        # self.video_tag = self.create_video_recorder(
        #     str(self.video_save_dir / "car_video_tag.avi")
        # )

        logging.info("Created a DeepPiCar")

    def init_cam(self):
        for _ in range(30):
            self.camera.read()

    def create_video_recorder(self, path):
        return cv2.VideoWriter(
            path,
            self.fourcc,
            self.fps,
            (int(self.camera.get(3)), int(self.camera.get(4))),
        )

    def warmup_obj_det(self):
        for _ in range(30):
            self.obj_det_model.interpreter.invoke()

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        if traceback is not None:
            logging.error("Exiting with statement with exception %s" % traceback)

        self.cleanup()

    def cleanup(self):
        """Reset the hardware"""
        logging.info("Stopping the car, resetting hardware.")
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.video_orig.release()
        self.camera.release()
        # self.camera.release()
        # self.video_lane.release()
        # self.video_objs.release()
        # self.video_tag.release()
        cv2.destroyAllWindows()

    def drive(self, speed=0):
        logging.info("Starting to drive at speed %s..." % speed)
        self.back_wheels.speed = speed

        while self.camera.isOpened():
            time.sleep(1e-9)
            _, image_org = self.camera.read()
            show_image("orig", image_org, self.show_image)

            # image_objs = image_org.copy()
            image_objs = self.obj_det_model.process_objects_on_road(image_org)
            # self.video_objs.write(image_objs)
            show_image("Detected Objects", image_objs, self.show_image)

            # image_lane = image_org.copy()
            # image_lane = self.follow_lane(image_lane)
            # self.video_lane.write(image_lane)
            # show_image("Lane Lines", image_lane, self.show_image)

            # image_tag = image_org.copy()
            # image_tag = self.process_tag(image_tag)
            # self.video_tag.write(image_tag)
            # show_image("April Tag", image_tag, self.show_image)

    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=0,
        help="Score threshold for detected objects",
    )
    parser.add_argument("--show_image", action="store_true")
    parser.set_defaults(show_image=False)

    args = parser.parse_args()

    speed = args.speed
    show_image = args.show_image

    with DeepPiCar(initial_speed=speed, show_image=show_image) as car:
        try:
            car.drive(speed)
        except KeyboardInterrupt:
            car.cleanup()
            print_statistics(car)
            sys.exit(0)

        print_statistics(car)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s:%(asctime)s.%(msecs)03d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
