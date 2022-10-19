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


def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen

    polygon = np.array(
        [
            [
                (0, height * 1 / 2),
                (width, height * 1 / 2),
                (width, height),
                (0, height),
            ]
        ],
        np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    # show_image("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(
        cropped_edges,
        rho,
        angle,
        min_threshold,
        np.array([]),
        minLineLength=8,
        maxLineGap=4,
    )

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def main():
    frame = cv2.imread("driver/data/road0.png")

    edges = detect_edges(frame)

    cropped_edges = region_of_interest(edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image(line_segment_image)


if __name__ == "__main__":
    main()
