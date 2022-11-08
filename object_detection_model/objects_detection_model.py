import cv2
import logging
import datetime
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from object_detection_model.traffic_objects import Person


class ObjectDetectionModel(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(
        self,
        car=None,
        speed_limit=40,
        model_path="experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite",
        label="object_detection_model/model/obj_det_labels.txt",
        width=320,
        height=180,
    ):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info("Creating a ObjectsOnRoadProcessor...")
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # initialize TensorFlow models
        self.labels = read_label_file(label)

        # initial edge TPU engine
        logging.info("Initialize Edge TPU with model %s..." % model_path)
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.min_confidence = 0.5
        self.num_of_objects = 3
        logging.info("Initialize Edge TPU with model done.")

        self.traffic_objects = {0: Person()}

        self.durations = []

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        start_time = time.perf_counter()
        logging.debug("Processing objects.................................")

        objects, final_frame = self.detect_objects(frame)

        duration = (time.perf_counter() - start_time) * 1000
        self.durations.append(duration)

        self.control_car(objects)

        logging.debug("Processing objects END.............................")

        return final_frame

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

        return objects, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def control_car(self, objects):
        logging.debug("Control car...")
        car_state = {"speed": self.speed_limit, "speed_limit": self.speed_limit}

        logging.info(f"{len(objects)} objects are found")
        for obj in objects:
            obj_label = self.labels[obj.id]
            processor = self.traffic_objects[obj.id]
            if processor.is_close_by(obj, self.height, min_height_pct=0):
                processor.set_car_state(car_state)
            else:
                logging.debug(
                    "[%s] object detected, but it is too far, ignoring. " % obj_label
                )
            self.resume_driving(car_state)

        # if len(objects) == 0:
        #     car_state["speed"] = self.speed_limit
        #     self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state["speed_limit"]
        self.speed = car_state["speed"]

        if self.speed == 0:
            self.set_speed(0)
        else:
            self.set_speed(self.speed_limit)
        logging.debug("Current Speed = %d, New Speed = %d" % (old_speed, self.speed))

        if self.speed == 0:
            logging.debug("full stop for 1 seconds")

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-5s:%(asctime)s: %(message)s"
    )
