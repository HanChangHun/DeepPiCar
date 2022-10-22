import time
import logging

import cv2
import picar
import numpy as np
from keras.models import load_model


class EndToEndLaneFollower(object):
    def __init__(
        self,
        model_path="lane_navigation/model/lane_navigation_final.h5",
    ):
        self.__SCREEN_WIDTH = 320
        self.__SCREEN_HEIGHT = 180
        self.show=False

        logging.info("Creating a EndToEndLaneFollower...")

        picar.setup()

        logging.debug("Set up camera")
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.back_wheels.speed = 0
        self.curr_steering_angle = 90
        self.model = load_model(model_path)

    def init_cam(self):
        for _ in range(50):
            _, image_lane = self.camera.read()
            show_image("Lane Lines", image_lane)

    def __exit__(self, _type, value, traceback):
        """Exit a with statement"""
        if traceback is not None:
            # Exception occurred:
            logging.error("Exiting with statement with exception %s" % traceback)

        self.cleanup()

    def drive(self, speed=0):
        logging.info("Starting to drive at speed %s..." % speed)
        self.back_wheels.speed = speed
        i = 0
        while self.camera.isOpened():
            time.sleep(1e-4)

            _, image_lane = self.camera.read()

            image_lane = self.follow_lane(image_lane)
            show_image("Lane Lines", image_lane, self.show)
            if self.show:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.cleanup()
                    break

    def follow_lane(self, frame):
        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("curr_steering_angle = %d" % self.curr_steering_angle)

        self.front_wheels.turn(self.curr_steering_angle)

    def compute_steering_angle(self, frame):
        """Find the steering angle directly based on video frame
        We assume that camera is calibrated to point to dead center
        """
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = self.model.predict(X)[0]

        logging.debug("new steering angle: %s" % steering_angle)
        return int(steering_angle + 0.5)  # round the nearest integer

    def cleanup(self):
        """Reset the hardware"""
        logging.info("Stopping the car, resetting hardware.")
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        cv2.destroyAllWindows()


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[
        int(height / 2) :, :, :
    ]  # remove top half of the image, as it is not relevant for lane following
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2YUV
    )  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    image = (
        image / 255
    )  # normalizing, the processed image becomes black for some reason.  do we need this?
    return image


def show_image(title, frame, show=False):
    if show:
        cv2.imshow(title, frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    lane_follower = EndToEndLaneFollower()
    lane_follower.init_cam()
    lane_follower.drive(15)
