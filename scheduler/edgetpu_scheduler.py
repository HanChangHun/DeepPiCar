from model_runner.model_runner import ModelRunner


class EdgeTPUScheduler:
    def __init__(self):
        self.waiting_queue = []

        self.model_runners = {}
        self.periods = {}
        self.task_results = {}

    def add_model_runner(self, task_name, period, segment_paths):
        model_runner = ModelRunner(segment_paths)
        model_runner.allocate_tensors_all_interpreter()
        model_runner.invoke_all(None)

        self.model_runners[task_name] = model_runner
        self.periods[task_name] = period
        self.task_results[task_name] = []

    def execute_request(self, task=None):
        if self.waiting_queue == []:
            return "empty", None

        self.sort_waiting_queue()
        cur_req = self.get_next_request()
        result = self.invoke_model_runner(cur_req, task)
        self.sort_waiting_queue()
        return result

    def sort_waiting_queue(self):
        self.waiting_queue.sort(
            key=lambda x: (self.periods[x["task_name"]], x["request_time"])
        )

    def get_next_request(self):
        cur_req = self.waiting_queue.pop(0)
        return cur_req

    def invoke_model_runner(self, request, task):
        task_name = request["task_name"]
        image = request["data"]

        model_runner: ModelRunner = self.model_runners[task_name]

        is_done = model_runner.invoke_and_next(image, task)

        if is_done:
            result = model_runner.get_result(task=task)
            return "complete", {"request": request, "result": result}

        else:
            self.reappend_request(request)
            return "incomplete", None

    def reappend_request(self, request):
        self.waiting_queue.append(request)
