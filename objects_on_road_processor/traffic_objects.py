from threading import Timer
import logging


class TrafficObject(object):
    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        # default: if a sign is 10% of the height of frame
        print(obj.bbox)
        rect_size = (obj.bbox.xmax - obj.bbox.xmin) * (obj.bbox.ymax - obj.bbox.ymin)
        return rect_size >= 350


class Person(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug("pedestrian: stopping car")
        car_state["speed"] = 0
