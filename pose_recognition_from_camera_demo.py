# -*- coding: utf-8 -*-
import cv2 as cv
import mediapipe as mp
import sys


def draw_landmarks(frame, landmarks, visibility_threshold=0.5):
    w, h = frame.shape[1], frame.shape[0]

    landmark_point = []

    for idx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * w), w - 1)
        landmark_y = min(int(landmark.y * h), h - 1)
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_threshold:
            continue

        if idx == 0:   # 鼻
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 1:   # 右目：上眼睑
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 2:   # 右目：眼瞳
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 3:   # 右目：下眼睑
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 4:   # 左目：上眼睑
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 5:   # 左目：眼瞳
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 6:   # 左目：下眼睑
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 7:   # 右耳
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 8:   # 左耳
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 9:   # 口：右端嘴角
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 10:  # 口：左端嘴角
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 11:  # 右肩
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 12:  # 左肩
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 13:  # 右肘
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 14:  # 左肘
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 15:  # 右手首
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 16:  # 左手首
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 17:  # 右手: 外側端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 18:  # 左手: 外側端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 19:  # 右手: 先端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 20:  # 左手: 先端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 21:  # 右手: 内側端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 22:  # 左手: 内側端
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 23:  # 腰: 右側
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 24:  # 腰: 左側
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 25:  # 右膝
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 26:  # 左膝
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 27:  # 右足首
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 28:  # 左足首
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 29:  # 右足跟
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 30:  # 左足跟
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 31:  # 右足
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if idx == 32:  # 左足
            cv.circle(frame, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

    # 右目
    if landmark_point[1][0] > visibility_threshold and landmark_point[2][0] > visibility_threshold:
        cv.line(frame, landmark_point[1][1], landmark_point[2][1], (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_threshold and landmark_point[3][0] > visibility_threshold:
        cv.line(frame, landmark_point[2][1], landmark_point[3][1], (0, 255, 0), 2)

    # 左目
    if landmark_point[4][0] > visibility_threshold and landmark_point[5][0] > visibility_threshold:
        cv.line(frame, landmark_point[4][1], landmark_point[5][1], (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_threshold and landmark_point[6][0] > visibility_threshold:
        cv.line(frame, landmark_point[5][1], landmark_point[6][1], (0, 255, 0), 2)

    # 口
    if landmark_point[9][0] > visibility_threshold and landmark_point[10][0] > visibility_threshold:
        cv.line(frame, landmark_point[9][1], landmark_point[10][1], (0, 255, 0), 2)

    # 肩
    if landmark_point[11][0] > visibility_threshold and landmark_point[12][0] > visibility_threshold:
        cv.line(frame, landmark_point[11][1], landmark_point[12][1], (0, 255, 0), 2)

    # 右腕
    if landmark_point[11][0] > visibility_threshold and landmark_point[13][0] > visibility_threshold:
        cv.line(frame, landmark_point[11][1], landmark_point[13][1],(0, 255, 0), 2)
    if landmark_point[13][0] > visibility_threshold and landmark_point[15][0] > visibility_threshold:
        cv.line(frame, landmark_point[13][1], landmark_point[15][1], (0, 255, 0), 2)

    # 左腕
    if landmark_point[12][0] > visibility_threshold and landmark_point[14][0] > visibility_threshold:
        cv.line(frame, landmark_point[12][1], landmark_point[14][1], (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_threshold and landmark_point[16][0] > visibility_threshold:
        cv.line(frame, landmark_point[14][1], landmark_point[16][1], (0, 255, 0), 2)

    # 右手
    if landmark_point[15][0] > visibility_threshold and landmark_point[17][0] > visibility_threshold:
        cv.line(frame, landmark_point[15][1], landmark_point[17][1], (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_threshold and landmark_point[19][0] > visibility_threshold:
        cv.line(frame, landmark_point[17][1], landmark_point[19][1], (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_threshold and landmark_point[21][0] > visibility_threshold:
        cv.line(frame, landmark_point[19][1], landmark_point[21][1], (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_threshold and landmark_point[15][0] > visibility_threshold:
        cv.line(frame, landmark_point[21][1], landmark_point[15][1], (0, 255, 0), 2)

    # 左手
    if landmark_point[16][0] > visibility_threshold and landmark_point[18][0] > visibility_threshold:
        cv.line(frame, landmark_point[16][1], landmark_point[18][1], (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_threshold and landmark_point[20][0] > visibility_threshold:
        cv.line(frame, landmark_point[18][1], landmark_point[20][1], (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_threshold and landmark_point[22][0] > visibility_threshold:
        cv.line(frame, landmark_point[20][1], landmark_point[22][1], (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_threshold and landmark_point[16][0] > visibility_threshold:
        cv.line(frame, landmark_point[22][1], landmark_point[16][1], (0, 255, 0), 2)

    # 胴体
    if landmark_point[11][0] > visibility_threshold and landmark_point[23][0] > visibility_threshold:
        cv.line(frame, landmark_point[11][1], landmark_point[23][1], (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_threshold and landmark_point[24][0] > visibility_threshold:
        cv.line(frame, landmark_point[12][1], landmark_point[24][1], (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_threshold and landmark_point[24][0] > visibility_threshold:
        cv.line(frame, landmark_point[23][1], landmark_point[24][1], (0, 255, 0), 2)

    if len(landmark_point) > 25:
        # 右足
        if landmark_point[23][0] > visibility_threshold and landmark_point[25][0] > visibility_threshold:
            cv.line(frame, landmark_point[23][1], landmark_point[25][1],
                    (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_threshold and landmark_point[27][0] > visibility_threshold:
            cv.line(frame, landmark_point[25][1], landmark_point[27][1],
                    (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_threshold and landmark_point[29][0] > visibility_threshold:
            cv.line(frame, landmark_point[27][1], landmark_point[29][1],
                    (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_threshold and landmark_point[31][0] > visibility_threshold:
            cv.line(frame, landmark_point[29][1], landmark_point[31][1],
                    (0, 255, 0), 2)

        # 左足
        if landmark_point[24][0] > visibility_threshold and landmark_point[26][0] > visibility_threshold:
            cv.line(frame, landmark_point[24][1], landmark_point[26][1],
                    (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_threshold and landmark_point[28][0] > visibility_threshold:
            cv.line(frame, landmark_point[26][1], landmark_point[28][1],
                    (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_threshold and landmark_point[30][0] > visibility_threshold:
            cv.line(frame, landmark_point[28][1], landmark_point[30][1],
                    (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_threshold and landmark_point[32][0] > visibility_threshold:
            cv.line(frame, landmark_point[30][1], landmark_point[32][1],
                    (0, 255, 0), 2)
    return frame


if __name__ == "__main__":
    # mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv.VideoCapture("./1629019814638682.mp4")
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
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame, exiting ...")
                break

            # Flip the frame horizontally for a later selfie-view display.
            frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the frame as 
            # not writeable to pass by reference.
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            # Draw the pose annotation on the frame.
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = draw_landmarks(frame, results.pose_landmarks)
            cv.imshow("pose_recognition_from_camera_demo", frame)
            if cv.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv.destroyAllWindows()
