import sys
import time
import json
import logging
import datetime
import argparse
from pathlib import Path

import cv2
import picar

from drive.utils import print_statistics, show_image
from drive.webcam_video_stream import WebcamVideoStream
from drive.video_recoder import VideoRecoder

from object_detection_model.objects_detection_model import ObjectDetectionModel


class DeepPiCar:
    def __init__(
        self, speed_limit=0, show_image=False, video_save_dir=Path("drive/data")
    ):
        """Init camera and wheels"""
        logging.info("Creating a DeepPiCar...")
        self.speed_limit = speed_limit
        self.show_image = show_image
        picar.setup()

        logging.debug("Set up camera")
        self.screen_width = 854
        self.screen_height = 480
        self.fps = 10.0

        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.out_dir = Path(video_save_dir / f"{date_str}")
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.camera = WebcamVideoStream(
            -1, self.screen_width, self.screen_height, self.fps
        ).start()

        logging.debug("Set up back wheels")
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        logging.debug("Set up front wheels")
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 5  # calibrate servo to center
        self.front_wheels.turn(
            90
        )  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        logging.debug("Set up object detection model")
        self.obj_det_model_path = (
            "experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite"
        )
        self.obj_det_model = ObjectDetectionModel(
            self,
            model_path=self.obj_det_model_path,
            speed_limit=self.speed_limit,
            width=self.screen_width,
            height=self.screen_height,
        )
        self.warmup_obj_det()

        logging.debug("Set up video stream")
        self.video_recoder = VideoRecoder(
            self.camera, self.out_dir / "video.avi", self.fps
        ).start()

        logging.info("Created a DeepPiCar")

        self.obj_results = []

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
        self.video_recoder.release()
        self.camera.release()
        with open(self.out_dir / "objects.json", "w") as f:
            json.dump(self.obj_results, f, indent=4)
        cv2.destroyAllWindows()

    def drive(self, speed=0):
        logging.info("Starting to drive at speed %s..." % speed)
        self.back_wheels.speed = speed

        while self.camera.isOpened():
            time.sleep(1e-9)
            _, frame = self.camera.read()
            show_image("orig", frame, self.show_image)

            objects, frame_obj = self.obj_det_model.process_objects_on_road(frame)
            obj_result = {"frame_cnt": self.video_recoder.frame_cnt, "objects": objects}
            self.obj_results.append(obj_result)
            show_image("Detected Objects", frame_obj, self.show_image)

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

    with DeepPiCar(speed_limit=speed, show_image=show_image) as car:
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
