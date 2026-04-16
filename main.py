import cv2
import threading
import sys
import os
import math

sys.path.append(os.path.dirname(__file__))

from utils.pose_detector       import create_pose, extract_features, draw_landmarks, draw_label
from utils.exercise_classifier import load_model, auto_detect_exercise, get_feedback, predict
from utils.audio_feedback      import speak
from chatbot                   import ask_fitbot

# ── Load ML model ─────────────────────────────────────────────────────────────
model, accuracy = load_model("models/poseguard_model.pkl")
if model is None:
    print("ERROR: Run train_model.py first!")
    exit()

# ── Chatbot state ─────────────────────────────────────────────────────────────
chat_lines      = [("FitBot", "Hi! Ask me anything about workouts!")]
input_text      = ""
is_loading      = False
chatbot_reply   = [None]
show_chatbot    = False

# ── App state ──────────────────────────────────────────────────────────────────
auto_mode       = True
mode            = "squat"

MANUAL_MODES    = {
    ord('1'): "squat",
    ord('2'): "bicep_curl",
    ord('3'): "plank",
    ord('4'): "lunge"
}

# ── Rep Counter State ─────────────────────────────────────────────────────────
rep_count = 0
rep_stage = None

# ── Angle Calculation ─────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

# ── Rep Counting ──────────────────────────────────────────────────────────────
def update_rep_count(landmarks, exercise, h, w):
    global rep_count, rep_stage

    def pt(i):
        lm = landmarks[i]
        return (int(lm.x*w), int(lm.y*h))

    LS, LE, LW = 11, 13, 15
    LH, LK, LA = 23, 25, 27

    RS, RE, RW = 12, 14, 16
    RH, RK, RA = 24, 26, 28

    if exercise == "bicep_curl":
        angle = (calculate_angle(pt(LS), pt(LE), pt(LW)) +
                 calculate_angle(pt(RS), pt(RE), pt(RW))) / 2

        if angle > 150:
            rep_stage = "down"
        if angle < 50 and rep_stage == "down":
            rep_stage = "up"
            rep_count += 1

    elif exercise == "squat":
        angle = (calculate_angle(pt(LH), pt(LK), pt(LA)) +
                 calculate_angle(pt(RH), pt(RK), pt(RA))) / 2

        if angle > 160:
            rep_stage = "up"
        if angle < 100 and rep_stage == "up":
            rep_stage = "down"
            rep_count += 1

    elif exercise == "lunge":
        angle = min(
            calculate_angle(pt(LH), pt(LK), pt(LA)),
            calculate_angle(pt(RH), pt(RK), pt(RA))
        )

        if angle > 150:
            rep_stage = "up"
        if angle < 100 and rep_stage == "up":
            rep_stage = "down"
            rep_count += 1

    return rep_count, rep_stage

# ── Streamlit Frame Function ─────────────────────────────────────────────────
streamlit_pose = create_pose()

def process_poseguard_frame(frame):
    global auto_mode, mode

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = streamlit_pose.process(rgb)
    rgb.flags.writeable = True

    exercise_name = mode.upper()
    rep_val = 0
    status_text = "WAITING"

    if results.pose_landmarks:
        draw_landmarks(frame, results)

        try:
            features = extract_features(results.pose_landmarks.landmark, h, w)

            if auto_mode:
                mode = auto_detect_exercise(features)

            prediction, conf = predict(model, features)
            feedback, color, status = get_feedback(prediction, mode)

            reps, stage = update_rep_count(results.pose_landmarks.landmark, mode, h, w)

            exercise_name = mode.upper()
            rep_val = reps
            status_text = status

            draw_label(frame, f"Exercise: {exercise_name}", (10, 40), (255,255,255))
            draw_label(frame, f"Reps: {rep_val}", (10, 70), (0,255,255))
            draw_label(frame, f"Status: {status}", (10, 100), color)

        except:
            pass

    return frame, exercise_name, rep_val, status_text

# ── Chatbot UI ───────────────────────────────────────────────────────────────
def draw_chatbot(frame, chat_lines, input_text, is_loading):
    h, w = frame.shape[:2]
    bx = w - 360
    by = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (w-10, by+260), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    y = by + 30
    for speaker, msg in chat_lines[-6:]:
        cv2.putText(frame, f"{speaker}: {msg}", (bx+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,255,200), 1)
        y += 20

# ── Main Loop ────────────────────────────────────────────────────────────────
def main():
    global auto_mode, mode, show_chatbot, input_text, is_loading

    cap = cv2.VideoCapture(0)
    speak("PoseGuard started")

    with create_pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed, exercise, reps, status = process_poseguard_frame(frame)

            cv2.imshow("PoseGuard", processed)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()