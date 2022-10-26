import logging

import numpy as np

from keras.models import load_model
from pycoral.utils.edgetpu import make_interpreter

from deep_pi_car.utils import show_image
from lane_navigation.utils import display_heading_line, img_preprocess, predict_steer


_SHOW_IMAGE = False


class LaneFollower(object):
    def __init__(
        self,
        car=None,
        model_path="lane_navigation/model/lane_navigation_w_pretrain_final.h5",
    ):
        self.car = car
        self.model = load_model(model_path)

    def follow_lane(self, frame):
        show_image("orig", frame, _SHOW_IMAGE)

        self.curr_steering_angle = self.compute_steering_angle(frame)
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


class LaneFollowerEdgeTPU(object):
    def __init__(
        self,
        car=None,
        model_path="lane_navigation/model/lane_navigation_w_pretrain_final_edgetpu.tflite",
    ):
        self.car = car

        self.model = make_interpreter(model_path)
        self.model.allocate_tensors()
        self.model.invoke()

    def follow_lane(self, frame):
        show_image("orig", frame, _SHOW_IMAGE)

        self.curr_steering_angle = self.compute_steering_angle(frame)
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
