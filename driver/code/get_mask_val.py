import cv2
import numpy as np

_SHOW_IMAGE = True


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # show_image("hsv", hsv)

    lower_black = np.array([0, 30, 50])
    upper_black = np.array([179, 100, 255])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    show_image("lane mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


def main():
    frame = cv2.imread("driver/data/road0.png")
    detect_edges(frame)


if __name__ == "__main__":
    main()
