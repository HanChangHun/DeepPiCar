# import the necessary packages
import time
from threading import Thread

import cv2


class VideoRecoder:
    def __init__(self, camera, out_path, fps=10):
        self.camera = camera
        self.out_path = str(out_path)
        # initialize the video camera stream and read the first frame
        # from the stream
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")

        self.video = self.create_video_recorder()

        self.frame_cnt = 0
        self.stopped = False

    def create_video_recorder(self):
        return cv2.VideoWriter(
            self.out_path,
            self.fourcc,
            self.fps,
            (int(self.camera.get(3)), int(self.camera.get(4))),
        )

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            time.sleep(1e-9)
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.video.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.camera.read()
            self.video.write(self.frame)
            self.frame_cnt += 1

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
