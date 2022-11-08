import logging
import time

import numpy as np

from pycoral.utils.edgetpu import make_interpreter

from lane_navigation_model.utils import (
    display_heading_line,
    img_preprocess,
    predict_steer,
)


class LaneNavigationModelEdgeTPU(object):
    def __init__(
        self,
        car=None,
        model_path="lane_navigation/model/lane_navigation_w_pretrain_final_edgetpu.tflite",
        show_image=False,
    ):
        self.car = car
        self.show_image = show_image

        self.model = make_interpreter(model_path)
        self.model.allocate_tensors()
        self.model.invoke()

        self.durations = []

    def follow_lane(self, frame):
        start_time = time.perf_counter()
        self.curr_steering_angle = self.compute_steering_angle(frame)
        duration = (time.perf_counter() - start_time) * 1000
        self.durations.append(duration)

        logging.debug(f"curr_steering_angle = {self.curr_steering_angle}")

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

    def compute_steering_angle(self, frame):
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = int(predict_steer(self.model, X)[0] + 0.5)

        return steering_angle
