from threading import Timer
import logging


class TrafficObject(object):
    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        # default: if a sign is 10% of the height of frame
        obj_height = obj.bounding_box[1][1] - obj.bounding_box[0][1]
        return obj_height / frame_height > min_height_pct


class Person(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug("pedestrian: stopping car")

        car_state["speed"] = 0
