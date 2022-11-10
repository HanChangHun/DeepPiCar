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

from drive.webcam_video_stream import WebcamVideoStream


def set_resized_input(interpreter, size, resize):
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = interpreter.tensor(interpreter.get_input_details()[0]["index"])()[0]
    tensor.fill(0)
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = np.reshape(result, (h, w, channel))

    cur_det_scale = (scale, scale)
    return cur_det_scale


def e2e_inference(interpreter, camera, image_path):

    st1 = int(time.perf_counter() * 1000)

    # image = cv2.imread(image_path)
    _, frame = camera.read()
    image = frame

    st2 = int(time.perf_counter() * 1000)

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_RGB)

    st3 = int(time.perf_counter() * 1000)

    scale = set_resized_input(
        interpreter,
        img_pil.size,
        lambda size: img_pil.resize(size, Image.Resampling.LANCZOS),
    )

    st4 = int(time.perf_counter() * 1000)

    interpreter.invoke()

    st5 = int(time.perf_counter() * 1000)

    objects = detect.get_objects(interpreter, score_threshold=0.3, image_scale=scale)

    st6 = int(time.perf_counter() * 1000)

    print(f"st1 {st2-st1} st2 {st3-st2} st3 {st4-st3} st4 {st5-st4} st5 {st6-st5} st6")


if __name__ == "__main__":
    __SCREEN_WIDTH = 854
    __SCREEN_HEIGHT = 480

    camera = cv2.VideoCapture(-1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    camera.set(cv2.CAP_PROP_FPS, 10.0)
    camera.set(3, __SCREEN_WIDTH)
    camera.set(4, __SCREEN_HEIGHT)

    model_path = "experiments/obj_det_sram/models/full/efficientdet-lite_edgetpu.tflite"

    image_path = "data/object_detection/test/frame_000004.JPEG"

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    interpreter.invoke()

    time_spans = []
    for _ in range(50):
        st = time.perf_counter()
        interpreter.invoke()
        dur = round((time.perf_counter() - st) * 1000, 2)
        time_spans.append(dur)
        print(f"inference {dur}")
    print("Average inference time: %f ms" % (sum(time_spans) / len(time_spans)))
    print(f"Std inference time: {np.std(time_spans)} ms\n")

    time_spans = []
    for _ in range(50):
        st = time.perf_counter()
        e2e_inference(interpreter, camera, image_path)
        time_spans.append((time.perf_counter() - st) * 1000)
    print("Average e2e time: %f ms" % (sum(time_spans) / len(time_spans)))
    print(f"Std e2e time: {np.std(time_spans)} ms\n")

    time_spans = []
    for _ in range(25):
        st = time.perf_counter()
        _, image = camera.read()
        time_spans.append((time.perf_counter() - st) * 1000)
    print("Average camera read time: %f ms" % (sum(time_spans) / len(time_spans)))
    print(f"Std camera read time: {np.std(time_spans)} ms")

    camera.release()
