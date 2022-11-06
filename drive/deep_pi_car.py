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

from lane_navigation.lane_follower_edgetpu import LaneFollowerEdgeTPU
from drive.utils import print_statistics, show_image


from objects_on_road_processor.objects_on_road_processor import ObjectsOnRoadProcessor

obj_det_model_path = (
    "objects_on_road_processor/model/ssd_mobilenet_v2/ssd_mobilenet_v2_edgetpu.tflite"
)


class DeepPiCar(object):
    def __init__(self, initial_speed=0, show_image=False):
        """Init camera and wheels"""
        logging.info("Creating a DeepPiCar...")
        self.show_image = show_image

        picar.setup()
        self.__SCREEN_WIDTH = 640
        self.__SCREEN_HEIGHT = 360

        self.initial_speed = initial_speed

        logging.debug("Set up camera")
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)
        self.init_cam()

        logging.debug("Set up back wheels")
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        logging.debug("Set up front wheels")
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0  # calibrate servo to center
        self.front_wheels.turn(
            90
        )  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        self.lane_follower = LaneFollowerEdgeTPU(
            self,
            model_path="co_compiled_model/lane_navigation_w_pretrain_final_edgetpu.tflite",
        )
        self.traffic_sign_processor = ObjectsOnRoadProcessor(
            self,
            model_path=obj_det_model_path,
            speed_limit=self.initial_speed,
            width=self.__SCREEN_WIDTH,
            height=self.__SCREEN_HEIGHT,
        )

        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_save_dir = Path(f"drive/data/{date_str}")
        self.video_save_dir.mkdir(exist_ok=True, parents=True)

        self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        self.video_orig = self.create_video_recorder(
            str(self.video_save_dir / "car_video.avi")
        )
        self.video_lane = self.create_video_recorder(
            str(self.video_save_dir / "car_video_lane.avi")
        )
        self.video_objs = self.create_video_recorder(
            str(self.video_save_dir / "car_video_objs.avi")
        )
        self.video_tag = self.create_video_recorder(
            str(self.video_save_dir / "car_video_tag.avi")
        )

        with open("calibrationValues0.json") as f:
            cal_vals = json.load(f)
        self.cam_mtx = np.array(cal_vals["camera_matrix"])
        self.distor_factor = np.array(cal_vals["dist_coeff"])

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

        logging.info("Created a DeepPiCar")

    def create_video_recorder(self, path):
        return cv2.VideoWriter(
            path, self.fourcc, 20.0, (int(self.camera.get(3)), int(self.camera.get(4)))
        )

    def init_cam(self):
        for _ in range(50):
            _, image_lane = self.camera.read()
            show_image("Lane Lines", image_lane, self.show_image)

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
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()

    def drive(self, speed=0):
        logging.info("Starting to drive at speed %s..." % speed)
        self.back_wheels.speed = speed
        while self.camera.isOpened():
            _, image_org = self.camera.read()
            show_image("orig", image_org, self.show_image)
            self.video_orig.write(image_org)

            image_objs = image_org.copy()
            image_objs = self.process_objects_on_road(image_objs)
            self.video_objs.write(image_objs)
            show_image("Detected Objects", image_objs, self.show_image)

            # image_lane = image_org.copy()
            # image_lane = self.follow_lane(image_lane)
            # self.video_lane.write(image_lane)
            # show_image("Lane Lines", image_lane, self.show_image)

            image_tag = image_org.copy()
            image_tag = self.process_tag(image_tag)
            self.video_tag.write(image_tag)
            show_image("April Tag", image_tag, self.show_image)

            if show_image:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.cleanup()
                    break

    def process_objects_on_road(self, image):
        image = self.traffic_sign_processor.process_objects_on_road(image)
        return image

    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image

    def process_tag(self, image):
        marker_length = 0.1
        corners, ids, _ = aruco.detectMarkers(image, self.aruco_dict)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.1, self.cam_mtx, self.distor_factor
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


def distance_to_camera(pixel_width):
    knownWidth = 15
    focalLength = 160

    return (knownWidth * focalLength) / pixel_width


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
