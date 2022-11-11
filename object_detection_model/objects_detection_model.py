import cv2
import time
import json
import logging
from threading import Thread

import numpy as np
from PIL import Image


from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from scheduler.edgetpu_scheduler import EdgeTPUScheduler

from object_detection_model.traffic_objects import Person


class ObjectDetectionModel(object):
    def __init__(
        self,
        car,
        task_name,
        period,
        segment_paths,
        scheduler: EdgeTPUScheduler,
        speed_limit=40,
        label="object_detection_model/model/obj_det_labels.txt",
    ):
        logging.info("Creating a ObjectDetectionModel...")
        self.car = car
        self.task_name = task_name
        self.period = period
        self.segment_paths = segment_paths
        self.scheduler = scheduler
        self.speed_limit = speed_limit
        self.speed = speed_limit
        self.lock = scheduler.lock

        self.labels = read_label_file("data/object_detection/labels.txt")

        self.traffic_objects = {0: Person()}
        self.min_confidence = 0.5

        self.objects = None
        self.stopped = False

    def start(self):
        self.register_model()
        self.send_request()
        self.recieve_result()
        return self

    def read(self):
        return self.objects

    def release(self):
        self.stopped = True

    def register_model(self):
        self.scheduler.add_model_runner(
            self.task_name, "detection", self.period, self.segment_paths
        )

    def send_request(self):
        def send_request_loop(start_time):
            _iter = 0
            while True:
                time.sleep(1e-9)
                if self.stopped:
                    return

                if time.perf_counter() - start_time > self.period * _iter:
                    _, frame = self.car.camera.read()
                    self.lock.acquire()
                    self.scheduler.waiting_queue.append(
                        {
                            "task_name": self.task_name,
                            "task": "detection",
                            "request_time": time.perf_counter(),
                            "data": frame,
                        }
                    )
                    self.lock.release()
                    _iter += 1

        start_time = time.perf_counter()
        send_request_th = Thread(target=send_request_loop, args=(start_time,))
        send_request_th.start()

    def recieve_result(self):
        def recieve_result_loop():
            while True:
                time.sleep(1e-4)
                if self.stopped:
                    return

                self.objects = self.scheduler.task_results[self.task_name]
                self.control_car(self.objects)

        send_request_th = Thread(target=recieve_result_loop)
        send_request_th.start()

    def control_car(self, objects):
        logging.debug("Control car...")
        car_state = {
            "speed": self.speed_limit,
            "speed_limit": self.speed_limit,
        }

        logging.debug(f"{len(objects)} objects are found")
        for obj in objects:
            if obj != []:
                obj_label = self.labels[obj.id]
                processor = self.traffic_objects[obj.id]
                if processor.is_close_by(obj):
                    processor.set_car_state(car_state)
                else:
                    logging.debug(
                        "[%s] object detected, but it is too far, ignoring. "
                        % obj_label
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
        logging.debug(
            "Current Speed = %d, New Speed = %d" % (old_speed, self.speed)
        )

        if self.speed == 0:
            logging.debug("full stop for 1 seconds")

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed

    def update(self):
        start_time = time.perf_counter()
        _iter = 0

        while True:
            time.sleep(1e-9)
            if self.stopped:
                return

            if time.perf_counter() - start_time > self.period * _iter:
                logging.info(f"iteration {_iter}")
                _, frame = self.car.camera.read()
                self.objects = self.process_objects_on_road(frame)

                _iter += 1

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug("Processing objects..........................")

        start_time = time.perf_counter()
        objects = self.detect_objects(frame)
        duration = (time.perf_counter() - start_time) * 1000

        logging.debug(
            f"Processing objects END. Duration: {round(duration, 2)}...."
        )
        self.durations.append(duration)

        self.control_car(objects)

        return objects

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
            self.interpreter,
            score_threshold=self.min_confidence,
            image_scale=scale,
        )

        return objects


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-5s:%(asctime)s: %(message)s"
    )
