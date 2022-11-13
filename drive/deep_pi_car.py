import sys
import time
import json
import logging
import datetime
import argparse
from pathlib import Path
from threading import Lock

import cv2
import picar

from drive.utils import print_statistics, show_image
from drive.webcam_video_stream import WebcamVideoStream
from drive.video_recoder import VideoRecoder
from scheduler.edgetpu_scheduler import EdgeTPUScheduler

from object_detection_model.objects_detection_model import ObjectDetectionModel
from interference_model.interference_model import InterferenceModel


class DeepPiCar:
    def __init__(
        self,
        speed_limit=0,
        show_image=False,
        video_save_dir=Path("drive/data"),
    ):

        """Init camera and wheels"""
        logging.info("Creating a DeepPiCar...")
        self.speed_limit = speed_limit
        self.show_image = show_image
        picar.setup()

        logging.info("Set up video stream and recoder")
        screen_width = 854
        screen_height = 480
        fps = 10.0

        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.out_dir = Path(video_save_dir / f"{date_str}")
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.camera = WebcamVideoStream(-1, screen_width, screen_height, fps).start()
        self.video_recoder = VideoRecoder(
            self.camera, self.out_dir / "video.avi", fps
        ).start()

        logging.info("Set up back wheels")
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0 # Speed Range is 0 (stop) - 100 (fastest)
        self.back_wheels.backward()

        logging.info("Set up front wheels")
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 9  # calibrate servo to center
        # Steering Range is 45 (left) - 90 (center) - 135 (right)
        self.front_wheels.turn(90)

        logging.info("Set up Edge TPU scheduler")
        self.lock = Lock()
        self.edgetpu_scheduler = EdgeTPUScheduler(self.lock).start()

        logging.info("Set up object detection model")
        det_period = 0.5
        # ours
        obj_det_model_paths = [
            "experiments/co_compile_obj_cls/model/ours/efficientdet-lite_edgetpu.tflite"
        ]
        # baseline
        obj_det_model_paths = [
            "experiments/co_compile_obj_cls/model/baseline/efficientdet-lite_edgetpu.tflite"
        ]
        self.obj_det_model = ObjectDetectionModel(
            self,
            "efficientdet-lite0",
            det_period,
            obj_det_model_paths,
            self.edgetpu_scheduler,
            self.speed_limit,
        ).start()

        logging.info("Set up interference classification model")
        cls_period = 2.0
        # ours
        cls_segment_paths = [
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_0_of_6_edgetpu.tflite",
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_1_of_6_edgetpu.tflite",
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_2_of_6_edgetpu.tflite",
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_3_of_6_edgetpu.tflite",
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_4_of_6_edgetpu.tflite",
            "experiments/co_compile_obj_cls/model/ours/segmented/inception_v2_224_quant/inception_v2_224_quant_segment_5_of_6_edgetpu.tflite",
        ]
        # baseline
        cls_segment_paths = [
            "experiments/co_compile_obj_cls/model/baseline/inception_v2_224_quant_edgetpu.tflite"
        ]

        self.cls_model = InterferenceModel(
            self,
            "efficientnet-M",
            cls_period,
            cls_segment_paths,
            self.edgetpu_scheduler,
        ).start()

        logging.info("Created a DeepPiCar")

        self.obj_results = []

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
        self.cls_model.release()
        self.obj_det_model.release()
        self.edgetpu_scheduler.release()
        self.camera.release()
        with open(self.out_dir / "objects.json", "w") as f:
            json.dump(self.obj_results, f, indent=4)
        cv2.destroyAllWindows()

    def drive(self, speed=0):
        time.sleep(5)
        logging.info("Starting to drive at speed %s..." % speed)
        self.back_wheels.speed = speed

        # self.edgetpu_scheduler.init_logs()

        while self.camera.isOpened():
            time.sleep(1e-4)
            # _, frame = self.camera.read()
            # show_image("orig", frame, self.show_image)

            objects = self.obj_det_model.read()
            obj_result = {
                "frame_cnt": self.video_recoder.frame_cnt,
                "objects": objects,
            }
            self.obj_results.append(obj_result)
            # show_image("Detected Objects", frame_obj, self.show_image)


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
            # print_statistics(car)
            sys.exit(0)

        # print_statistics(car)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s:%(asctime)s.%(msecs)03d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
