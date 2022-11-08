import cv2
import logging
import datetime
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file


def e2e_inference(interpreter, image_path):
    image = cv2.imread(image_path)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_RGB)
    _, scale = common.set_resized_input(
        interpreter,
        img_pil.size,
        lambda size: img_pil.resize(size, Image.ANTIALIAS),
    )

    interpreter.invoke()
    objects = detect.get_objects(interpreter, score_threshold=0.3, image_scale=scale)


if __name__ == "__main__":
    model_path = "experiments/obj_det_sram/models/zero/efficientdet-lite_edgetpu.tflite"

    image_path = "data/object_detection/test/frame_000004.JPEG"

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    interpreter.invoke()

    time_spans = []
    for _ in range(20):
        st = time.perf_counter()
        interpreter.invoke()
        time_spans.append((time.perf_counter() - st) * 1000)

    print("Average inference time: %f ms" % (sum(time_spans) / len(time_spans)))

    time_spans = []
    for _ in range(20):
        st = time.perf_counter()
        e2e_inference(interpreter, image_path)
        time_spans.append((time.perf_counter() - st) * 1000)

    print("Average e2e time: %f ms" % (sum(time_spans) / len(time_spans)))
