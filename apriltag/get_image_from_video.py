import cv2


def main():
    video_path = "drive/data/221027_205558/car_video.avi"
    out_image_path = "apriltag/temp.jpg"

    vidcap = cv2.VideoCapture(video_path)
    if vidcap.isOpened():
        for _ in range(85):
            _, _ = vidcap.read()
        success, image = vidcap.read()
        cv2.imwrite(out_image_path, image)
    vidcap.release()


if __name__ == "__main__":
    main()
