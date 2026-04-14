import cv2
import threading
import sys
import os

sys.path.append(os.path.dirname(__file__))

from utils.pose_detector      import create_pose, extract_features, draw_landmarks, draw_label
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
MANUAL_MODES    = {ord('1'): "squat", ord('2'): "bicep_curl",
                   ord('3'): "plank", ord('4'): "lunge"}

# ── Draw chatbot overlay ───────────────────────────────────────────────────────
def draw_chatbot(frame, chat_lines, input_text, is_loading):
    h, w  = frame.shape[:2]
    bx    = w - 375
    by    = 10
    bw    = 365
    bh    = 290

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 200, 100), 2)

    cv2.putText(frame, "FitBot AI  (C=toggle)", (bx+8, by+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 120), 1, cv2.LINE_AA)
    cv2.line(frame, (bx, by+28), (bx+bw, by+28), (0, 200, 100), 1)

    y_pos = by + 48
    for speaker, msg in chat_lines[-7:]:
        color = (180, 255, 180) if speaker == "You" else (210, 210, 255)
        prefix = f"{speaker}: "
        words  = msg.split()
        line   = ""
        lx     = bx + 8 + len(prefix) * 7

        cv2.putText(frame, prefix, (bx+8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)

        for word in words:
            if len(line + word) > 34:
                cv2.putText(frame, line.strip(), (lx, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)
                y_pos += 15
                lx     = bx + 16
                line   = word + " "
            else:
                line += word + " "
        if line.strip():
            cv2.putText(frame, line.strip(), (lx, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)
        y_pos += 18

        if y_pos > by + bh - 40:
            break

    if is_loading:
        cv2.putText(frame, "FitBot is thinking...", (bx+8, by+bh-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (bx, by+bh-24), (bx+bw, by+bh), (35, 35, 35), -1)
    cv2.putText(frame, "> " + input_text + "|", (bx+6, by+bh-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    global auto_mode, mode, show_chatbot, input_text, is_loading

    cap = cv2.VideoCapture(0)
    speak("Welcome to PoseGuard. Auto detection is on.")

    with create_pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # MediaPipe processing
            import cv2 as _cv2
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                draw_landmarks(frame, results)

                try:
                    features = extract_features(results.pose_landmarks.landmark, h, w)

                    if auto_mode:
                        mode = auto_detect_exercise(features)

                    prediction, conf = predict(model, features)
                    feedback, color, status = get_feedback(prediction, mode)

                    tag = "AUTO" if auto_mode else "MANUAL"
                    draw_label(frame, f"PoseGuard  [{tag}]",        (10, 32),  (100, 220, 100))
                    draw_label(frame, f"Exercise : {mode.upper()}", (10, 64),  (255, 255, 255))
                    draw_label(frame, f"Status   : {status}",       (10, 96),  color)
                    draw_label(frame, f"Conf     : {conf:.1f}%",    (10, 128), (255, 255, 0))
                    draw_label(frame, feedback,                      (10, 160), color)
                    draw_label(frame, f"Model Acc: {accuracy*100:.1f}%", (10, 192), (180, 180, 180))

                    if status == "WRONG":
                        speak(feedback)

                except Exception:
                    draw_label(frame, "Stand in front of camera", (10, 96), (0, 165, 255))
            else:
                draw_label(frame, "No person detected", (10, 96), (0, 0, 255))

            # Controls hint
            cv2.putText(frame,
                        "A=Auto  1=Squat 2=Curl 3=Plank 4=Lunge  C=Chat  Q=Quit",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

            # Chatbot overlay
            if show_chatbot:
                if chatbot_reply[0]:
                    chat_lines.append(("FitBot", chatbot_reply[0]))
                    speak(chatbot_reply[0])
                    chatbot_reply[0] = None
                    is_loading       = False
                draw_chatbot(frame, chat_lines, input_text, is_loading)

            cv2.imshow("PoseGuard — AI Gym Posture Detector", frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_mode = not auto_mode
                speak("Auto mode on" if auto_mode else "Manual mode on")
            elif key in MANUAL_MODES:
                mode      = MANUAL_MODES[key]
                auto_mode = False
                speak(f"Switched to {mode.replace('_', ' ')}")
            elif key == ord('c'):
                show_chatbot = not show_chatbot
            elif show_chatbot:
                if key == 13 and input_text.strip() and not is_loading:
                    question       = input_text.strip()
                    chat_lines.append(("You", question))
                    input_text     = ""
                    is_loading     = True

                    def fetch(q):
                        chatbot_reply[0] = ask_fitbot(q)
                    threading.Thread(target=fetch, args=(question,), daemon=True).start()

                elif key == 8:
                    input_text = input_text[:-1]
                elif 32 <= key <= 126:
                    input_text += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    speak("PoseGuard session ended.")
    print("Session ended.")


if __name__ == "__main__":
    main()
