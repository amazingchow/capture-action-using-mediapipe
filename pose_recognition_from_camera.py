# -*- coding: utf-8 -*-
import argparse
import cv2 as cv
import math
import mediapipe as mp
import numpy as np
import sys
import time


def transform(image, landmarks, color=(0, 0, 0), bg_color=(255, 255, 255, 0), visibility_threshold=0.3):
    w, h = image.shape[1], image.shape[0]

    # set the background
    cv.rectangle(image, (0, 0), (w, h), bg_color, thickness=-1)

    landmark_infos = []
    for idx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * w), w - 1)
        landmark_y = min(int(landmark.y * h), h - 1)
        landmark_infos.append([idx, landmark.visibility, (landmark_x, landmark_y)])

    # set the face
    (face_center_x, face_center_y), face_radius = min_enclosing_face_circle(landmark_infos)
    face_center_x = int(face_center_x)
    face_center_y = int(face_center_y)
    face_radius = int(face_radius * 1.2)
    cv.circle(image, (face_center_x, face_center_y), face_radius, color, -1)
    # set the upper body
    upper_body_center_x = (landmark_infos[11][2][0] + landmark_infos[12][2][0] + \
        landmark_infos[24][2][0] + landmark_infos[23][2][0]) // 4
    upper_body_center_y = (landmark_infos[11][2][1] + landmark_infos[12][2][1] + \
        landmark_infos[24][2][1] + landmark_infos[23][2][1]) // 4
    axes_h = int(((landmark_infos[11][2][0] + landmark_infos[23][2][0]) / 2 - \
        (landmark_infos[12][2][0] + landmark_infos[24][2][0]) / 2) * 0.618)
    axes_v = int(((landmark_infos[23][2][1] + landmark_infos[24][2][1]) / 2 - \
        (landmark_infos[11][2][1] + landmark_infos[12][2][1]) / 2) * 0.618)
    cv.ellipse(image, (upper_body_center_x, upper_body_center_y), (axes_h, axes_v), 0, 0, 360, color, -1)
    # set left upper limb
    point_set = np.array([
        landmark_infos[11][2],
        landmark_infos[13][2],
        landmark_infos[15][2],
        ((landmark_infos[17][2][0] + landmark_infos[19][2][0]) // 2, \
            (landmark_infos[17][2][1] + landmark_infos[19][2][1]) // 2),
    ])
    for p1, p2 in zip(point_set, point_set[1:]):
        cv.line(image, p1, p2, color, 2)
    # set right upper limb
    point_set = np.array([
        landmark_infos[12][2],
        landmark_infos[14][2],
        landmark_infos[16][2],
        ((landmark_infos[18][2][0] + landmark_infos[20][2][0]) // 2, \
            (landmark_infos[18][2][1] + landmark_infos[20][2][1]) // 2),
    ])
    for p1, p2 in zip(point_set, point_set[1:]):
        cv.line(image, p1, p2, color, 2)
    # set left leg
    point_set = np.array([
        landmark_infos[23][2],
        landmark_infos[25][2],
        landmark_infos[27][2],
        ((landmark_infos[29][2][0] + landmark_infos[31][2][0]) // 2, \
            (landmark_infos[29][2][1] + landmark_infos[31][2][1]) // 2),
    ])
    for p1, p2 in zip(point_set, point_set[1:]):
        cv.line(image, p1, p2, color, 2)
    # set right leg
    point_set = np.array([
        landmark_infos[24][2],
        landmark_infos[26][2],
        landmark_infos[28][2],
        ((landmark_infos[30][2][0] + landmark_infos[32][2][0]) // 2, \
            (landmark_infos[30][2][1] + landmark_infos[32][2][1]) // 2),
    ])
    for p1, p2 in zip(point_set, point_set[1:]):
        cv.line(image, p1, p2, color, 2)

    return image



def min_enclosing_face_circle(landmark_infos):
    point_set = np.empty((0, 2), int)

    for idx in [2, 5, 7, 8, 9, 10]:
        landmark_point = [
            np.array((landmark_infos[idx][2][0], landmark_infos[idx][2][1]))
        ]
        point_set = np.append(point_set, landmark_point, axis=0)

    return cv.minEnclosingCircle(points=point_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device",             type=int,   default=0)
    parser.add_argument("--video_file",               type=str,   default="")
    parser.add_argument("--video_output_file",        type=str,   default="./output.mp4")
    parser.add_argument("--video_output_width",       type=int,   default=0)
    parser.add_argument("--video_output_height",      type=int,   default=0)
    parser.add_argument("--static_image_mode",        type=bool,  default=False)
    parser.add_argument("--model_complexity",         type=int,   default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    args = parser.parse_args()

    cap = object()
    out = object()
    if args.video_file != "":
        cap = cv.VideoCapture(args.video_file)
    else:
        cap = cv.VideoCapture(args.video_device)
    if not cap.isOpened():
        print("Cannot open camera device")
        sys.exit(-1)
    else:
        w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CAP_PROP_FPS)
        print("Video <width: {}, height: {}, fps: {}>".format(w, h, fps))
        if args.video_output_width > 0 and args.video_output_height > 0:
            w = args.video_output_width
            h = args.video_output_height
        if args.video_file != "":
            out = cv.VideoWriter(args.video_output_file, apiPreference=0, 
                fourcc=cv.VideoWriter_fourcc(*'MJPG'), fps=fps, frameSize=(int(w), int(h)))

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:

        image = np.zeros((int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 3), np.uint8)
        frame_idx = 0
        while 1:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Cannot receive frame, exiting ...")
                break
            frame_idx += 1

            st = time.time()
            frame_rgb = cv.cvtColor(cv.flip(frame_bgr, 1), cv.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if results.pose_landmarks is not None:
                image = transform(image, results.pose_landmarks)
            ed = time.time()
            print("Used {:.3f} secs to process frame-{:05}".format(ed - st, frame_idx))

            cv.imshow("Animation", image)
            if args.video_file != "":
                out.write(image)
            if cv.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        if args.video_file != "":
            out.release()
        cv.destroyAllWindows()
