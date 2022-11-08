import logging
import time

import numpy as np

from keras.models import load_model

from lane_navigation_model.utils import display_heading_line, img_preprocess, show_image


_SHOW_IMAGE = False


class LaneNavigationModel(object):
    def __init__(
        self,
        car=None,
        model_path="lane_navigation/model/lane_navigation_w_pretrain_final.h5",
    ):
        self.car = car
        self.model = load_model(model_path)
        self.durations = []

    def follow_lane(self, frame):
        show_image("orig", frame, _SHOW_IMAGE)

        start_time = time.perf_counter()
        self.curr_steering_angle = self.compute_steering_angle(frame)
        duration = time.perf_counter() - start_time
        self.durations.append(duration)
        logging.debug(f"curr_steering_angle = {self.curr_steering_angle}")

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

    def compute_steering_angle(self, frame):
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = int(self.model.predict(X)[0] + 0.5)

        return steering_angle
