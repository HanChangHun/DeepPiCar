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

from objects_on_road_processor.traffic_objects import Person

_SHOW_IMAGE = False


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(
        self,
        car=None,
        speed_limit=40,
        model="objects_on_road_processor/model/efficientdet-lite_edgetpu.tflite",
        label="objects_on_road_processor/model/obj_det_labels.txt",
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
        logging.info("Initialize Edge TPU with model %s..." % model)
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()

        self.min_confidence = 0.3
        self.num_of_objects = 3
        logging.info("Initialize Edge TPU with model done.")

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

        self.traffic_objects = {0: Person()}

        self.durations = []

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug("Processing objects.................................")
        start_time = time.perf_counter()
        objects, final_frame = self.detect_objects(frame)
        duration = time.perf_counter() - start_time
        self.durations.append(duration)
        self.control_car(objects)
        logging.debug("Processing objects END.............................")

        return final_frame

    def control_car(self, objects):
        logging.debug("Control car...")
        car_state = {"speed": self.speed_limit, "speed_limit": self.speed_limit}

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
            time.sleep(1)

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed

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
        scale_factor = 320 / img_pil.width

        draw_objects(ImageDraw.Draw(img_pil), objects, scale_factor, self.labels)

        return objects, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def draw_objects(draw, objs, scale_factor, labels):
    """Draws the bounding box and label for each object."""
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-5s:%(asctime)s: %(message)s"
    )
