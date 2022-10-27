# import the necessary packages
import apriltag
import argparse
import cv2


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth 


def init_cam(camera):
    for _ in range(50):
        _, image_lane = camera.read()


def main():
    KNOWN_WIDTH = 15
    KNOWN_DISTANCE =30 
    focalLength = 160

    camera = cv2.VideoCapture(-1)
    camera.set(3, 320)
    camera.set(4, 180)
    init_cam(camera)

    _, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define the AprilTags detector options and then detect the AprilTags
    # in the input image
    print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 0, 255), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        cv2.circle(image, ptA, 5, (0, 0, 255), -1)
        
        # draw the tag family on the image
        pixel_width = pow(pow((ptB[0] - ptA[0]), 2) + pow((ptB[1] - ptA[1]), 2), 0.5)
        distance = distance_to_camera(KNOWN_WIDTH, focalLength, pixel_width)
        print(distance)

        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(
            image,
            # tagFamily,
            str(round(distance,2)),
            (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    cv2.imwrite("Image.jpg", image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    camera.release()


    # focalLength = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print(focalLength)


if __name__ == "__main__":
    main()
