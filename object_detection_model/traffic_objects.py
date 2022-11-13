from threading import Timer
import logging


class TrafficObject(object):
    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj):
        # default: if a sign is 10% of the height of frame
        bbox = obj.bbox
        rect_size = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
        return rect_size >= 2000


class Person(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug("pedestrian: stopping car")
        car_state["speed"] = 0
        car_state["speed_limit"] = 0
