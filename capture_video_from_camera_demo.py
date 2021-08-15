# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera device")
        sys.exit(-1)
    else:
        print("video width: {}, height: {}, fps: {}".format(
            cap.get(cv.CAP_PROP_FRAME_WIDTH),
            cap.get(cv.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv.CAP_PROP_FPS)
        ))

    while 1:
        # read frame one-by-one
        ret, frame = cap.read()
        # if frame is read correctly, ret is will be True
        if not ret:
            print("Can't receive frame, exiting ...")
            break

        gray_im = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("capture_video_from_camera_demo", gray_im)
        if cv.waitKey(5) & 0xFF == 27:
            break

    # when everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
