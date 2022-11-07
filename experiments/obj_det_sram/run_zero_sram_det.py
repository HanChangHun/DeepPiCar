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


if __name__ == "__main__":
    model_path = "experiments/obj_det_sram/models/zero/efficientdet-lite_edgetpu.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    interpreter.invoke()

    time_spans = []
    for _ in range(100):
        st = time.perf_counter()
        interpreter.invoke()
        time_spans.append((time.perf_counter() - st) * 1000)

    print("Average time: %f ms" % (sum(time_spans) / len(time_spans)))
