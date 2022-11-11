from pprint import pprint
import time
import collections

import cv2
import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters.detect import BBox
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

labels = read_label_file("data/object_detection/labels.txt")


class ModelRunner:
    def __init__(self, model_paths):
        self.model_paths = model_paths

        self.interpreters: list[tflite.Interpreter] = []
        self.make_interpreters()

        self.intermediate = dict()
        self.cur_idx = 0
        self.cur_det_scale = None

        output_details = self.interpreters[-1].get_output_details()[0]
        self.scale, self.zero_point = output_details["quantization"]

        self.size = common.input_size(self.interpreters[0])
        _dtype = self.interpreters[0].get_input_details()[0]["dtype"]

    def make_interpreters(self):
        for model_path in self.model_paths:
            self.interpreters.append(make_interpreter(str(model_path)))

    def allocate_tensors_all_interpreter(self):
        for interpreter in self.interpreters:
            interpreter.allocate_tensors()

    def invoke_all(self, image, task):
        # this function will be executed for warmup interpreter
        self.cur_idx = 0

        for _ in range(len(self.interpreters)):
            self.invoke_and_next(image, task)

    def invoke_and_next(self, image, task):
        assert self.cur_idx < len(self.interpreters)

        self.invoke_idx(self.cur_idx, image, task)

        if self.cur_idx < len(self.interpreters) - 1:
            self.cur_idx += 1
            return 0

        elif self.cur_idx == len(self.interpreters) - 1:
            self.cur_idx = 0
            return 1

        else:
            raise Exception("wired index...")

    def invoke_idx(self, idx, image, task, profile=False):
        interpreter = self.interpreters[idx]

        if image is not None:
            h2d_dur = self.set_input(idx, image, task, profile=profile)

        exec_dur = self.invoke(interpreter, profile=profile)
        d2h_dur = self.update_intermediate(interpreter, profile=profile)

        if profile:
            return [h2d_dur, exec_dur, d2h_dur]

    def set_input(self, idx, image, task, profile=False):
        if idx == 0:
            duration = self.set_first_input(image, task, profile=profile)
        else:
            duration = self.set_general_input(
                self.interpreters[idx], profile=profile
            )

        if profile:
            return duration

    def set_first_input(self, image, task, profile=False):
        if profile:
            st = time.perf_counter()

        if task == "classification":
            pass
            # process for single frame image
            # image = np.resize(image, [*self.size, 3])[np.newaxis, :]

            # input_details = self.interpreters[0].get_input_details()
            # for input_detail in input_details:
            #     if input_detail["name"] == "images":
            #         tensor_index = input_detail["index"]
            #         self.interpreters[0].set_tensor(tensor_index, image)

        elif task == "detection":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            self.set_resized_input(
                self.interpreters[0],
                image.size,
                lambda size: image.resize(size, Image.ANTIALIAS),
            )

        else:
            raise Exception("unsupported task")

        if profile:
            duration = (time.perf_counter() - st) * 1000
            return duration

    def set_resized_input(self, interpreter, size, resize):
        _, height, width, _ = interpreter.get_input_details()[0]["shape"]
        w, h = size
        scale = min(width / w, height / h)
        w, h = int(w * scale), int(h * scale)
        tensor = interpreter.tensor(
            interpreter.get_input_details()[0]["index"]
        )()[0]
        tensor.fill(0)
        _, _, channel = tensor.shape
        result = resize((w, h))
        tensor[:h, :w] = np.reshape(result, (h, w, channel))

        self.cur_det_scale = (scale, scale)

    def set_general_input(self, interpreter, profile=False):
        assert self.intermediate != None
        if profile:
            st = time.perf_counter()

        input_details = interpreter.get_input_details()
        # print("---")
        # pprint(input_details)
        print("---")
        pprint(list(self.intermediate.keys()))
        print("---")
        for input_detail in input_details:
            for k, v in self.intermediate.items():
                if input_detail["name"] == k:
                    interpreter.set_tensor(input_detail["index"], v)
        if profile:
            duration = (time.perf_counter() - st) * 1000
            return duration

    def invoke(self, interpreter, profile=False):
        if profile:
            st = time.perf_counter()

        interpreter.invoke()

        if profile:
            duration = (time.perf_counter() - st) * 1000
            return duration

    def update_intermediate(self, interpreter, profile=False):
        if profile:
            st = time.perf_counter()

        for o in interpreter.get_output_details():
            self.intermediate[o["name"]] = interpreter.get_tensor(o["index"])

        if profile:
            duration = (time.perf_counter() - st) * 1000
            return duration

    def get_result(self, task, top_n=1, thres=0.5):
        out = list(self.intermediate.items())
        # assert len(out) == 1

        _, values = out[0]
        result = None
        if task == "classification":
            scores = self.scale * (
                values[0].astype(np.int64) - self.zero_point
            )
            classes = classify.get_classes_from_scores(scores, top_n, 0.0)
            result = {labels.get(c.id, c.id): c.score for c in classes}

        elif task == "detection":
            result = self.get_objects(thres=thres)

        self.intermediate = dict()

        return result

    def get_objects(self, thres=0.5):
        Object = collections.namedtuple("Object", ["id", "score", "bbox"])

        def make(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return Object(
                id=int(class_ids[i]),
                score=float(scores[i]),
                bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                .scale(sx, sy)
                .map(int),
            )

        interpreter = self.interpreters[-1]
        signature_list = interpreter._get_full_signature_list()
        # pylint: enable=protected-access
        if signature_list:
            if len(signature_list) > 1:
                raise ValueError("Only support model with one signature.")
            signature = signature_list[next(iter(signature_list))]
            count = int(
                interpreter.tensor(signature["outputs"]["output_0"])()[0]
            )
            scores = interpreter.tensor(signature["outputs"]["output_1"])()[0]
            class_ids = interpreter.tensor(signature["outputs"]["output_2"])()[
                0
            ]
            boxes = interpreter.tensor(signature["outputs"]["output_3"])()[0]
        elif common.output_tensor(interpreter, 3).size == 1:
            boxes = common.output_tensor(interpreter, 0)[0]
            class_ids = common.output_tensor(interpreter, 1)[0]
            scores = common.output_tensor(interpreter, 2)[0]
            count = int(common.output_tensor(interpreter, 3)[0])
        else:
            scores = common.output_tensor(interpreter, 0)[0]
            boxes = common.output_tensor(interpreter, 1)[0]
            count = (int)(common.output_tensor(interpreter, 2)[0])
            class_ids = common.output_tensor(interpreter, 3)[0]

        width, height = self.size
        image_scale_x, image_scale_y = self.cur_det_scale
        sx, sy = width / image_scale_x, height / image_scale_y
        self.cur_det_scale = None

        return [make(i) for i in range(count) if scores[i] >= thres]
