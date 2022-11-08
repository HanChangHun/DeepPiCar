import sys
import time

from pathlib import Path
import cv2

import getch
import picar
from drive.webcam_video_stream import WebcamVideoStream

# back_wheels.forward()
# back_wheels.backward()


class KeypressDrive:
    def __init__(self) -> None:
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0

        picar.setup()

        self.video_stream = WebcamVideoStream(-1).start()

        self.cur_angle = 90
        self.cur_speed = 0
        self.step = 1
        self.previous_key = None
        self.idx = 0

        data_dir = Path("train_data_generation/data/drive_with_keypress")
        self.lab_cnt = len(list(data_dir.glob("*")))
        self.lab_dir = data_dir / f"{self.lab_cnt + 1}"
        self.lab_dir.mkdir(exist_ok=True, parents=True)

    def init_car(self):
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.cur_angle = 90

    def cleanup(self):
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.video_stream.stop()
        cv2.destroyAllWindows()

    def set_speed(self, speed):
        self.back_wheels.speed = speed

    def write_data(self):
        image = self.video_stream.read()
        cv2.imwrite(
            str(self.lab_dir / f"frame_{self.idx:06}_{int(self.cur_angle + 0.5)}.JPEG"),
            image,
        )
        self.idx += 1

    def go_front(self):
        self.cur_speed += 30
        self.set_speed(self.cur_speed)
        time.sleep(0.5)
        self.cur_speed -= 30
        self.set_speed(self.cur_speed)

    def steer_left(self):
        if self.previous_key == "a":
            self.step *= 1.3
        else:
            self.step = 1
        self.steer(-1 * self.step)
        self.previous_key = "a"

    def steer_right(self):
        if self.previous_key == "d":
            self.step *= 1.3
        else:
            self.step = 1
        self.steer(2 * self.step)
        self.previous_key = "d"

    def steer(self, angle):
        self.cur_angle += angle

        if self.cur_angle < 45:
            self.cur_angle = 45
        elif self.cur_angle > 135:
            self.cur_angle = 135

        self.front_wheels.turn(self.cur_angle)
        # print(f"current angle: {self.cur_angle}")
        # self.write_data()

    def start(self):
        self.init_car()
        self.set_speed(self.cur_speed)
        print("start keypress drving")
        while True:
            key = getch.getch()
            if key == "w":
                self.go_front()

            if key == "a":
                self.steer_left()

            if key == "d":
                self.steer_right()

            if key == "c":
                self.write_data()

            if key == "x":
                break

        self.cleanup()


if __name__ == "__main__":
    keypress_drive = KeypressDrive()
    try:
        keypress_drive.start()
    except KeyboardInterrupt:
        keypress_drive.cleanup()
        sys.exit(0)
