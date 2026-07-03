# PoseGuard

An AI-powered gym posture detection system that watches your form in real time through a webcam and tells you when you're doing it wrong — before you hurt yourself.

---

## The Problem

Most people at the gym have no idea their form is off. A personal trainer costs money and isn't always available. Bad form over time leads to injuries that sideline you for months. PoseGuard is the always-on form checker that doesn't need a trainer in the room.

---

## What it does

Point your webcam at yourself while working out. PoseGuard detects which exercise you're doing, analyzes your body angles using MediaPipe, classifies your posture as correct or incorrect using a trained ML model, and speaks the correction out loud if something's off. There's also a FitBot chatbot for workout questions.

Supported exercises: squats, bicep curls, planks, lunges.

---

## Tech Stack

- Python
- MediaPipe BlazePose — body keypoint detection
- OpenCV — video processing
- Random Forest Classifier — posture classification
- pyttsx3 — text-to-speech audio feedback
- Groq API (LLaMA3) — FitBot AI chatbot
- Streamlit — dashboard UI

---

## Model Performance

Trained on 100,000+ samples across 4 exercises.

| Metric | Score |
|--------|-------|
| Test Accuracy | 86% |
| Macro Avg F1 | 89% |
| Algorithm | Random Forest (200 trees) |

---

## Project Structure

```
PoseGuard/
├── main.py
├── app.py
├── chatbot.py
├── train_model.py
├── utils/
│   ├── pose_detector.py
│   ├── exercise_classifier.py
│   └── audio_feedback.py
├── screenshots/
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/mayankkhadse/PoseGuard.git
cd PoseGuard
pip install -r requirements-local.txt
```

Set your Groq API key:
```bash
# Windows
setx GROQ_API_KEY "your_api_key_here"
```

Train the model:
```bash
py -3.11 train_model.py
```

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

Or run the raw detection:
```bash
py -3.11 main.py
```

---

## Controls (main.py mode)

| Key | Action |
|-----|--------|
| A | Toggle auto / manual exercise detection |
| 1 | Squat mode |
| 2 | Bicep curl mode |
| 3 | Plank mode |
| 4 | Lunge mode |
| C | Open / close FitBot chatbot |
| Q | Quit |

---

## Note on deployment

This app is designed to run locally — it needs access to your webcam and speakers. Remote hosting on Streamlit Cloud won't give you the full experience.

---

## What's next

- Support for more exercises (deadlift, push-up, shoulder press)
- Rep counter with set tracking
- Session summary with posture score
- Mobile support via phone camera

---

## Author

Mayank Khadse — B.Tech Electronics & Telecommunication, Suryodaya College of Engineering and Technology, Nagpur
