# PoseGuard — AI Gym Posture Detection System

> Real-time AI + ML gym posture detection using MediaPipe and Random Forest. Detects 4 exercises with live audio feedback and an AI fitness chatbot.

---

## Features

- **Auto Exercise Detection** — automatically detects squat, bicep curl, plank, lunge from body angles
- **Real-time Posture Feedback** — classifies posture as correct or wrong using ML
- **Audio Feedback** — speaks corrections aloud when posture is wrong
- **FitBot AI Chatbot** — ask any workout question powered by Groq AI
- **86%+ Accuracy** — trained on real dataset of 100,000+ samples

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Body Detection | MediaPipe BlazePose |
| ML Model | Random Forest Classifier |
| Computer Vision | OpenCV |
| Audio | pyttsx3 |
| AI Chatbot | Groq API (LLaMA3) |
| Language | Python 3.11 |

---

## Exercises Supported

| Exercise | Detection Method |
|----------|----------------|
| Squat | Knee + hip angle from real image dataset |
| Bicep Curl | Elbow angle from CSV landmark data |
| Plank | Hip + shoulder alignment from CSV data |
| Lunge | Knee + hip angle from CSV landmark data |

---

## Project Structure

```
PoseGuard/
├── Dataset/
│   ├── squat/          ← image dataset (train/test)
│   ├── bicepcurl/      ← CSV landmark data
│   ├── plank/          ← CSV landmark data
│   └── lunge/          ← CSV landmark data
├── models/
│   └── poseguard_model.pkl  ← trained ML model
├── utils/
│   ├── pose_detector.py
│   ├── exercise_classifier.py
│   └── audio_feedback.py
├── main.py             ← run this to start
├── poseguard.py        ← single file version
├── chatbot.py          ← Groq AI chatbot
├── train_model.py      ← retrain the ML model
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/mayankkhadse/PoseGuard.git
cd PoseGuard
```

**2. Install dependencies:**
```bash
pip install opencv-python mediapipe==0.10.14 numpy scikit-learn pandas pyttsx3 groq
```

**3. Add your Groq API key in `chatbot.py`:**
```python
GROQ_API_KEY = "your_gsk_key_here"
```

**4. Train the model:**
```bash
py -3.11 train_model.py
```

**5. Run PoseGuard:**
```bash
py -3.11 main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| A | Toggle auto / manual mode |
| 1 | Squat mode |
| 2 | Bicep curl mode |
| 3 | Plank mode |
| 4 | Lunge mode |
| C | Open / close FitBot chatbot |
| Q | Quit |

---

## ML Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 86% |
| Macro Avg F1 | 89% |
| Training Samples | 100,000+ |
| Exercises | 4 |
| Algorithm | Random Forest (200 trees) |

---

## IEEE Paper

This project is documented as an IEEE format research paper:
`IEEE_Paper_Mayank_Khadse.docx`

---

## Author

**Mayank Khadse**
B-Tech Electronics & Telecommunication Engineering
Suryodaya College of Engineering and Technology (RTMNU), Nagpur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/mayank-khadse)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/mayankkhadse)
