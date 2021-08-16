# -*- coding: utf-8 -*-
import argparse
import cv2 as cv
import math
import mediapipe as mp
import numpy as np
import sys


def min_enclosing_face_circle(landmark_info):
    point_set = np.empty((0, 2), int)

    index_list = [2, 5, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array((landmark_info[index][2][0], landmark_info[index][2][1]))
        ]
        point_set = np.append(point_set, np_landmark_point, axis=0)

    return cv.minEnclosingCircle(points=point_set)


def draw(image, landmarks, color=(255, 255, 255), bg_color=(0, 0, 0), visibility_threshold=0.5):
    w, h = image.shape[1], image.shape[0]

    # 背景
    cv.rectangle(image, (0, 0), (w, h), bg_color, thickness=-1)

    landmark_infos = []
    for idx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * w), w - 1)
        landmark_y = min(int(landmark.y * h), h - 1)
        landmark_z = landmark.z
        landmark_infos.append([idx, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_infos)
    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)
    # 头
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    stick_radius_01 = int(face_radius * (4 / 5))
    stick_radius_02 = int(stick_radius_01 * (3 / 4))
    stick_radius_03 = int(stick_radius_02 * (3 / 4))        

    right_hip = landmark_infos[23]
    left_hip = landmark_infos[24]
    hip_x = int((right_hip[2][0] + left_hip[2][0]) / 2)
    hip_y = int((right_hip[2][1] + left_hip[2][1]) / 2)
    landmark_infos[23][2] = (hip_x, hip_y)
    landmark_infos[24][2] = (hip_x, hip_y)

    sorted_landmark_infos = sorted(landmark_infos, reverse=True, key=lambda x: x[3])

    draw_list = [
        11,  # 右腕
        12,  # 左腕
        23,  # 右脚
        24,  # 左脚
    ]

    # 腕/脚
    for landmark_info in sorted_landmark_infos:
        idx = landmark_info[0]
        if idx in draw_list:
            info_01 = [x for x in landmark_infos if x[0] == idx][0]
            info_02 = [x for x in landmark_infos if x[0] == (idx + 2)][0]
            info_03 = [x for x in landmark_infos if x[0] == (idx + 4)][0]

            if info_01[1] > visibility_threshold and info_02[1] > visibility_threshold:
                image = draw_stick(image, info_01[2], stick_radius_01, info_02[2], stick_radius_02, color)
            if info_02[1] > visibility_threshold and info_03[1] > visibility_threshold:
                image = draw_stick(image, info_02[2], stick_radius_02, info_03[2], stick_radius_03, color)

    return image


def draw_stick(image, p1, p1_radius, p2, p2_radius, color):
    cv.circle(image, p1, p1_radius, color, -1)
    cv.circle(image, p2, p2_radius, color, -1)

    draw_list = []
    for idx in range(2):
        rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        rad = rad + (math.pi / 2) + (math.pi * idx)
        p_x = int(p1_radius * math.cos(rad)) + p1[0]
        p_y = int(p1_radius * math.sin(rad)) + p1[1]
        draw_list.append([p_x, p_y])
        p_x = int(p2_radius * math.cos(rad)) + p2[0]
        p_y = int(p2_radius * math.sin(rad)) + p2[1]
        draw_list.append([p_x, p_y])
    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device",             type=int,   default=0)
    parser.add_argument("--video_file",               type=str,   default="")
    parser.add_argument("--video_output_file",        type=str,   default="./output.mp4")
    parser.add_argument("--video_width",              type=int,   default=0)
    parser.add_argument("--video_height",             type=int,   default=0)
    parser.add_argument("--static_image_mode",        type=bool,  default=False)
    parser.add_argument("--model_complexity",         type=int,   default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    args = parser.parse_args()

    # init video capture handler
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
        if args.video_width > 0 and args.video_height > 0:
            w = args.video_width
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            h = args.video_height
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        print("Video <w: {}, h: {}, fps: {}>".format(w, h, fps))
        out = cv.VideoWriter(args.video_output_file, apiPreference=0, 
            fourcc=cv.VideoWriter_fourcc(*'MJPG'), fps=fps, frameSize=(int(w), int(h)))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    image = np.zeros((int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 3), np.uint8)
    while 1:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame, exiting ...")
            break

        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = pose.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.pose_landmarks is not None:
            image = draw(image, results.pose_landmarks)
        cv.imshow("Animation", image)

        out.write(image)
        if cv.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
