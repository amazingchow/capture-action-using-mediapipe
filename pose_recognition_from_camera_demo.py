# -*- coding: utf-8 -*-
import argparse
import cv2 as cv
import mediapipe as mp
import sys
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", type=int, default=0)
    parser.add_argument("--video_file",   type=str, default="")
    args = parser.parse_args()
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = object()
        if args.video_file != "":
            cap = cv.VideoCapture(args.video_file)
        else:
            cap = cv.VideoCapture(args.video_device)
        if not cap.isOpened():
            print("Cannot open camera device-0")
            sys.exit(-1)
        else:
            print("Video <width: {}, height: {}, fps: {}>".format(
                cap.get(cv.CAP_PROP_FRAME_WIDTH),
                cap.get(cv.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv.CAP_PROP_FPS)
            ))
        
        fps = int(cap.get(cv.CAP_PROP_FPS))
        frame_idx = 0
        while 1:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame, exiting ...")
                break
            frame_idx += 1

            st = time.time()
            # flip the frame horizontally for a later selfie-view display
            frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
            # to improve performance, optionally mark the frame as not writeable to pass by reference
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            # draw the pose annotation on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ed = time.time()
            print("Used {:.3f} secs to process frame-{:05}".format(ed - st, frame_idx))
            
            gap = 1000//fps - int(1000 * (ed - st))
            if gap < 5:
                gap = 5

            cv.imshow("pose_recognition_from_camera_demo", frame)
            if cv.waitKey(gap) & 0xFF == 27:
                break
        
        cap.release()
        cv.destroyAllWindows()
