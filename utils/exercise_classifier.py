import pickle
import numpy as np


def load_model(path="models/poseguard_model.pkl"):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Model loaded! Classes: {data['classes']}")
        print(f"Model accuracy: {data['accuracy']*100:.1f}%")
        return data['model'], data['accuracy']
    except FileNotFoundError:
        print("Model not found! Run train_model.py first.")
        return None, 0.0


def auto_detect_exercise(features):
    lk  = features[0]
    rk  = features[1]
    le  = features[2]
    re  = features[3]
    lh  = features[4]
    rh  = features[5]
    lsh = features[6]
    rsh = features[7]

    avg_knee   = (lk + rk) / 2
    avg_elbow  = (le + re) / 2
    avg_hip    = (lh + rh) / 2
    avg_should = (lsh + rsh) / 2

    if avg_knee < 150 and avg_hip < 140:
        return "squat"
    elif avg_elbow < 130 and avg_knee > 150:
        return "bicep_curl"
    elif avg_hip > 150 and avg_knee > 155 and avg_elbow < 160:
        return "plank"
    elif avg_knee < 160 and avg_hip > 100 and avg_knee > 80:
        return "lunge"
    else:
        return "squat"


def get_feedback(prediction, mode):
    correct_map = {
        "squat":      "squat_correct",
        "bicep_curl": "bicep_curl_correct",
        "plank":      "plank_correct",
        "lunge":      "lunge_correct",
    }
    tips = {
        "squat":      "Bend knees more — go deeper!",
        "bicep_curl": "Curl higher for full range!",
        "plank":      "Keep your body straight!",
        "lunge":      "Keep front knee above ankle!",
    }

    if prediction == correct_map.get(mode):
        return "Perfect form! Keep going!", (0, 255, 0), "CORRECT"
    elif prediction.startswith(mode):
        return tips.get(mode, "Fix your form!"), (0, 0, 255), "WRONG"
    else:
        return f"Wrong posture for {mode.upper()}!", (0, 165, 255), "WRONG"


def predict(model, features):
    pred = model.predict([features])[0]
    conf = max(model.predict_proba([features])[0]) * 100
    return pred, conf
