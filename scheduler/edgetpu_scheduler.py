import time
from threading import Thread

from model_runner.model_runner import ModelRunner


class EdgeTPUScheduler:
    def __init__(self, lock):
        self.lock = lock
        self.waiting_queue = []

        self.model_runners = {}
        self.periods = {}
        self.task_results = {}

        self.stopped = False
        with open("scheduler/result/classification.log", "w") as f:
            f.write("")

        with open("scheduler/result/detection.log", "w") as f:
            f.write("")

    def start(self):
        self.start_execute_request()
        return self

    def release(self):
        self.stopped = True

    def add_model_runner(self, task_name, task, period, segment_paths):
        model_runner = ModelRunner(segment_paths)
        model_runner.allocate_tensors_all_interpreter()
        model_runner.invoke_all(None, task=task)

        self.model_runners[task_name] = model_runner
        self.periods[task_name] = period
        self.task_results[task_name] = []

    def start_execute_request(self):
        def execute_request_loop():
            while True:
                time.sleep(1e-9)
                if self.stopped:
                    return
                self.execute_request()

        exec_req_th = Thread(target=execute_request_loop)
        exec_req_th.start()

    def execute_request(self):
        if self.waiting_queue == []:
            return "empty", None

        self.sort_waiting_queue()
        cur_req = self.get_next_request()
        result = self.invoke_model_runner(cur_req)
        self.sort_waiting_queue()
        return result

    def sort_waiting_queue(self):
        sort_algo = "rm"
        sort_algo = "fifo"
        if sort_algo == "rm":
            self.lock.acquire()
            self.waiting_queue.sort(
                key=lambda x: (self.periods[x["task_name"]], x["request_time"])
            )
            self.lock.release()

    def get_next_request(self):
        self.lock.acquire()
        cur_req = self.waiting_queue.pop(0)
        self.lock.release()
        return cur_req

    def invoke_model_runner(self, request):
        task_name = request["task_name"]
        task = request["task"]
        data = request["data"]

        model_runner: ModelRunner = self.model_runners[task_name]

        is_done = model_runner.invoke_and_next(data, task)

        if is_done:
            result = model_runner.get_result(task=task)
            self.task_results[task_name] = result

            if task == "classification":
                with open("scheduler/result/classification.log", "a") as f:
                    f.write(
                        f"{time.perf_counter() - request['request_time']}\n"
                    )

            elif task == "detection":
                with open("scheduler/result/detection.log", "a") as f:
                    f.write(
                        f"{time.perf_counter() - request['request_time']}\n"
                    )

        else:
            self.reappend_request(request)

    def reappend_request(self, request):
        self.lock.acquire()
        self.waiting_queue.append(request)
        self.lock.release()
