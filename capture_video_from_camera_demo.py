# -*- coding: utf-8 -*-
import argparse
import cv2 as cv
import numpy as np
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", type=int, default=0)
    parser.add_argument("--video_file",   type=str, default="")
    args = parser.parse_args()

    cap = object()
    if args.video_file != "":
        cap = cv.VideoCapture(args.video_file)
    else:
        cap = cv.VideoCapture(args.video_device)
    if not cap.isOpened():
        print("Cannot open camera device-0")
        sys.exit(-1)
    else:
        print("OpenCV version: {}".format(cv.__version__))
        print("Video <width: {}, height: {}, fps: {}>".format(
            cap.get(cv.CAP_PROP_FRAME_WIDTH),
            cap.get(cv.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv.CAP_PROP_FPS)
        ))

    fps = int(cap.get(cv.CAP_PROP_FPS))
    while 1:
        # read frame one-by-one
        ret, frame = cap.read()
        # if frame is read correctly, ret will be True
        if not ret:
            print("Cannot receive frame, exiting ...")
            break

        gray_im = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("capture_video_from_camera_demo", gray_im)
        if cv.waitKey(1000//fps) & 0xFF == 27:
            break

    # when everything done, release the resource
    cap.release()
    cv.destroyAllWindows()
