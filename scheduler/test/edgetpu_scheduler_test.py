import unittest

import time
from threading import Thread, Lock

from drive.deep_pi_car import DeepPiCar
from interference_model.interference_model import InterferenceModel
from scheduler.edgetpu_scheduler import EdgeTPUScheduler


class TestEdgeTPUScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestEdgeTPUScheduler, cls).setUpClass()
        cls.test_start_time = time.time()

        cls.lock = Lock()
        cls.car = DeepPiCar()
        cls.edgetpu_scheduler = EdgeTPUScheduler(cls.lock)

        segment_paths = [
            "experiments/co_compile_obj_cls/model/efficientnet-M_edgetpu.tflite"
        ]

        period = 0.3
        cls.test_model = InterferenceModel(
            cls.car,
            "efficientnet-M",
            period,
            segment_paths,
            cls.edgetpu_scheduler,
            cls.lock,
        )
        cls.test_model.register_model()

    @classmethod
    def tearDownClass(cls):
        super(TestEdgeTPUScheduler, cls).tearDownClass()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_set_up(self):
        pass

    def test_execute_request(self):
        def execute_request_loop():
            while True:
                time.sleep(1e-9)
                if self.stop_threads:
                    return
                self.edgetpu_scheduler.execute_request()

        self.stop_threads = False
        exec_req_th = Thread(target=execute_request_loop)
        exec_req_th.start()
        self.test_model.start()

        test_timeout = 3
        time.sleep(test_timeout)

        self.test_model.release()
        self.stop_threads = True
        self.car.cleanup()


if __name__ == "__main__":
    unittest.main()
