# import the necessary packages
import time
from threading import Thread

import cv2


class WebcamVideoStream:
    def __init__(self, src=-1, screen_width=320, screen_height=180, fps=10):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(3, self.screen_width)
        self.stream.set(4, self.screen_height)

        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def get(self, num):
        return int(self.stream.get(num))

    def isOpened(self):
        return self.stream.isOpened()

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True

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
                self.stream.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame
