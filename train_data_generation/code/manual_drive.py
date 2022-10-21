import time
from pathlib import Path

import cv2
import picar
import numpy as np


back_wheels = picar.back_wheels.Back_Wheels()
front_wheels = picar.front_wheels.Front_Wheels()
front_wheels.offset = 20

picar.setup()

camera = cv2.VideoCapture(-1)
camera.set(3, 320)
camera.set(4, 180)


def init_car():
    back_wheels.speed = 0
    # back_wheels.forward()
    # back_wheels.backward()
    front_wheels.turn(90)


def go_front(speed, duration):
    back_wheels.speed = speed
    time.sleep(duration)
    back_wheels.speed = 0


def cleanup():
    back_wheels.speed = 0
    front_wheels.turn(90)
    camera.release()
    cv2.destroyAllWindows()


def write_data(idx):
    _, image = camera.read()
    cv2.imwrite(f"train_data_generation/data/frame_{idx}.png", image)


def main():
    init_car()
    for idx in range(10):
        go_front(20, 0.5)
        write_data(idx)
    cleanup()


if __name__ == "__main__":
    main()
