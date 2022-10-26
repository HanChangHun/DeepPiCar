import numpy as np
import cv2


def show_image(title, frame, show=True):
    if show:
        cv2.imshow(title, frame)


def print_statistics(car):
    print(
        f"lane navifation inference time mean: {np.mean(car.lane_follower.durations[1:])}"
    )
    print(
        f"lane navifation inference time std: {np.std(car.lane_follower.durations[1:])}"
    )
    print(
        f"lane navifation inference time fps: {1000 / np.mean(car.lane_follower.durations[1:])}"
    )
