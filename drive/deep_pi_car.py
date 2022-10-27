import sys
import logging
import datetime
import argparse
from pathlib import Path

import picar

import cv2

from lane_navigation.lane_follower_edgetpu import LaneFollowerEdgeTPU
from drive.utils import print_statistics, show_image


from objects_on_road_processor.objects_on_road_processor import ObjectsOnRoadProcessor


class DeepPiCar(object):
    def __init__(self, initial_speed=0, show_image=False):
        """Init camera and wheels"""
        logging.info("Creating a DeepPiCar...")
        self.show_image = show_image

        picar.setup()
        self.__SCREEN_WIDTH = 320
        self.__SCREEN_HEIGHT = 180
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

        # self.lane_follower = LaneFollowerEdgeTPU(
        #     self,
        #     show_image=self.show_image,
        # )
        # self.traffic_sign_processor = ObjectsOnRoadProcessor(
        #     self,
        #     speed_limit=self.initial_speed,
        #     show_image=self.show_image,
        # )

        self.lane_follower = LaneFollowerEdgeTPU(
            self,
            model_path="co_compiled_model/lane_navigation_w_pretrain_final_edgetpu.tflite",
            show_image=self.show_image,
        )
        self.traffic_sign_processor = ObjectsOnRoadProcessor(
            self,
            model_path="co_compiled_model/efficientdet-lite_edgetpu.tflite",
            speed_limit=self.initial_speed,
            show_image=self.show_image,
        )

        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_save_dir = Path(f"drive/data/{date_str}")
        self.video_save_dir.mkdir(exist_ok=True, parents=True)

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_orig = self.create_video_recorder(
            str(self.video_save_dir / "car_video.avi")
        )
        self.video_lane = self.create_video_recorder(
            str(self.video_save_dir / "car_video_lane.avi")
        )
        self.video_objs = self.create_video_recorder(
            str(self.video_save_dir / "car_video_objs.avi")
        )

        logging.info("Created a DeepPiCar")

    def create_video_recorder(self, path):
        return cv2.VideoWriter(
            path, self.fourcc, 20.0, (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT)
        )

    def init_cam(self):
        for _ in range(50):
            _, image_lane = self.camera.read()
            show_image("Lane Lines", image_lane)

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
        i = 0
        while self.camera.isOpened():
            _, image_lane = self.camera.read()
            show_image("orig", image_lane, self.show_image)
            image_objs = image_lane.copy()
            i += 1
            self.video_orig.write(image_lane)

            image_objs = self.process_objects_on_road(image_objs)
            self.video_objs.write(image_objs)
            show_image("Detected Objects", image_objs, self.show_image)

            image_lane = self.follow_lane(image_lane)
            self.video_lane.write(image_lane)
            show_image("Lane Lines", image_lane, self.show_image)

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
