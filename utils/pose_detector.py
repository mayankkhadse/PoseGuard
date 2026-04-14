import cv2
import mediapipe as mp
import numpy as np

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle   = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def extract_features(landmarks, h, w):
    def pt(idx):
        return [landmarks[idx].x * w, landmarks[idx].y * h]

    ls = pt(11); rs = pt(12)
    le = pt(13); re = pt(14)
    lw = pt(15); rw = pt(16)
    lh = pt(23); rh = pt(24)
    lk = pt(25); rk = pt(26)
    la = pt(27); ra = pt(28)

    return [
        calculate_angle(lh, lk, la),
        calculate_angle(rh, rk, ra),
        calculate_angle(ls, le, lw),
        calculate_angle(rs, re, rw),
        calculate_angle(ls, lh, lk),
        calculate_angle(rs, rh, rk),
        calculate_angle(le, ls, lh),
        calculate_angle(re, rs, rh),
    ]


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def draw_label(frame, text, pos, color):
    x, y = pos
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x-5, y-22), (x+tw+8, y+6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def create_pose():
    return mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
