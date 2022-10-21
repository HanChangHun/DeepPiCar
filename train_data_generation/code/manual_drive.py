import time
import json
from pathlib import Path

import cv2
import picar
import numpy as np

with open("train_data_generation/code/steer_angles.json", "r") as f:
    angles = json.load(f)

back_wheels = picar.back_wheels.Back_Wheels()
front_wheels = picar.front_wheels.Front_Wheels()
front_wheels.turning_offset = 0

picar.setup()

camera = cv2.VideoCapture(-1)
camera.set(3, 320)
camera.set(4, 180)


def init_car():
    back_wheels.speed = 0
    # back_wheels.forward()
    # back_wheels.backward()
    front_wheels.turn(90)


def init_cam():
    for _ in range(50):
        _, image = camera.read()


def go_front(speed, duration, turn=90):
    front_wheels.turn(turn)
    back_wheels.speed = speed
    time.sleep(duration)
    # back_wheels.speed = 0
    # time.sleep(0.1)


def cleanup():
    back_wheels.speed = 0
    front_wheels.turn(90)
    camera.release()
    cv2.destroyAllWindows()


def write_data(idx):
    _, image = camera.read()
    cv2.imwrite(
        f"train_data_generation/data/frame_{idx:06}_{round(angles[idx], 2)}.png", image
    )


def main():
    init_car()
    init_cam()
    for idx in range(len(angles)):
        # for idx in range(1):
        go_front(40, 0.01, angles[idx])
        write_data(idx)
    cleanup()


if __name__ == "__main__":
    main()
