import time
import logging
from threading import Thread

import numpy as np

from scheduler.edgetpu_scheduler import EdgeTPUScheduler


class InterferenceModel:
    def __init__(
        self,
        car,
        task_name,
        period,
        segment_paths,
        scheduler: EdgeTPUScheduler,
    ):
        logging.info("Creating a InterferenceModel...")
        self.car = car
        self.task_name = task_name
        self.period = period
        self.segment_paths = segment_paths
        self.scheduler = scheduler
        self.lock = scheduler.lock

        self.result = None
        self.stopped = False

    def start(self):
        self.register_model()
        self.send_request()
        self.recieve_result()
        return self

    def read(self):
        return self.result

    def release(self):
        self.stopped = True

    def register_model(self):
        self.scheduler.add_model_runner(
            self.task_name, "classification", self.period, self.segment_paths
        )

    def send_request(self):
        def send_request_loop(start_time):
            _iter = 0
            while True:
                time.sleep(1e-4)
                if self.stopped:
                    return

                if time.perf_counter() - start_time > self.period * _iter:
                    _, frame = self.car.camera.read()
                    self.lock.acquire()
                    self.scheduler.waiting_queue.append(
                        {
                            "task_name": self.task_name,
                            "task": "classification",
                            "request_time": time.perf_counter(),
                            "data": frame,
                        }
                    )
                    self.lock.release()
                    _iter += 1

        start_time = time.perf_counter()
        send_request_th = Thread(target=send_request_loop, args=(start_time,))
        send_request_th.start()

    def recieve_result(self):
        def recieve_result_loop():
            while True:
                time.sleep(1e-4)
                if self.stopped:
                    return

                self.result = self.scheduler.task_results[self.task_name]
                # print(self.result)

        send_request_th = Thread(target=recieve_result_loop)
        send_request_th.start()
