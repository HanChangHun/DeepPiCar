# import the necessary packages
import json
import math
import time
import argparse

import numpy as np
import cv2
import cv2.aruco as aruco


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def init_cam(camera):
    for _ in range(50):
        _, image_lane = camera.read()


def main():
    with open("calibrationValues0.json") as f:
        cal_vals = json.load(f)
    mtx = np.array(cal_vals["camera_matrix"])
    distor = np.array(cal_vals["dist_coeff"])

    image = cv2.imread("apriltag/temp.jpg")

    aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, distor)

    if ids is not None:
        for i in range(0, ids.size):
            aruco.drawAxis(image, mtx, distor, rvec[0], tvec[0], 0.06)
            cv2.putText(
                image,
                "%.1f cm -- %.0f degree"
                % ((tvec[0][0][2] * 100), (rvec[0][0][2] / math.pi * 180)),
                (0, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (244, 244, 244),
            )
            print((int)(tvec[0][0][2] * 1000))

    cv2.imwrite("apriltag/image2.jpg", image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # focalLength = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print(focalLength)


if __name__ == "__main__":
    main()
