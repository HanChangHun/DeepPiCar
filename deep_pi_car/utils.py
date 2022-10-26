import cv2


def show_image(title, frame, show=True):
    if show:
        cv2.imshow(title, frame)
