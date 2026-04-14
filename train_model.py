import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=True)

# ── Calculate angle between 3 points ─────────────────────────────────────────
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle   = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# ── Extract features from one CSV row ────────────────────────────────────────
def extract_features_from_row(row):
    def pt(name):
        x_col = f"{name}_x"
        y_col = f"{name}_y"
        if x_col in row.index and y_col in row.index:
            return [float(row[x_col]), float(row[y_col])]
        return [0.0, 0.0]

    ls = pt("left_shoulder");  rs = pt("right_shoulder")
    le = pt("left_elbow");     re = pt("right_elbow")
    lw = pt("left_wrist");     rw = pt("right_wrist")
    lh = pt("left_hip");       rh = pt("right_hip")
    lk = pt("left_knee");      rk = pt("right_knee")
    la = pt("left_ankle");     ra = pt("right_ankle")

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

# ── Load a CSV file ───────────────────────────────────────────────────────────
def load_csv(filepath, exercise):
    if not os.path.exists(filepath):
        print(f"  SKIP  : {filepath}")
        return [], []

    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
    except Exception as e:
        print(f"  ERROR : {filepath} — {e}")
        return [], []

    data, labels, skipped = [], [], 0

    for _, row in df.iterrows():
        try:
            features = extract_features_from_row(row)

            if 'label' in df.columns:
                raw = str(row['label']).strip().upper()
                label = f"{exercise}_correct" if raw in ['C','CORRECT','1','GOOD'] else f"{exercise}_wrong"
            else:
                label = f"{exercise}_correct"

            data.append(features)
            labels.append(label)
        except Exception:
            skipped += 1

    print(f"  LOADED: {filepath} → {len(data)} samples ({skipped} skipped)")
    return data, labels

# ── Load image folder (for squat) ─────────────────────────────────────────────
def load_images(folder, label):
    if not os.path.exists(folder):
        print(f"  SKIP  : {folder}")
        return [], []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]

    data, labels = [], []

    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            def pt(i): return [lm[i].x, lm[i].y]
            features = [
                calculate_angle(pt(23), pt(25), pt(27)),
                calculate_angle(pt(24), pt(26), pt(28)),
                calculate_angle(pt(11), pt(13), pt(15)),
                calculate_angle(pt(12), pt(14), pt(16)),
                calculate_angle(pt(11), pt(23), pt(25)),
                calculate_angle(pt(12), pt(24), pt(26)),
                calculate_angle(pt(13), pt(11), pt(23)),
                calculate_angle(pt(14), pt(12), pt(24)),
            ]
            data.append(features)
            labels.append(label)

    print(f"  LOADED: {folder} → {len(data)} images")
    return data, labels

# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading datasets...")
print("=" * 55)

all_data, all_labels = [], []

# BICEP CURL
print("\n[1] Bicep Curl (CSV)")
for f in ["Dataset/bicepcurl/train.csv",
          "Dataset/bicepcurl/test.csv",
          "Dataset/bicepcurl/evaluation.csv"]:
    d, l = load_csv(f, "bicep_curl")
    all_data += d; all_labels += l

# PLANK
print("\n[2] Plank (CSV)")
for f in ["Dataset/plank/train.csv",
          "Dataset/plank/test.csv",
          "Dataset/plank/evaluation.csv"]:
    d, l = load_csv(f, "plank")
    all_data += d; all_labels += l

# LUNGE
print("\n[3] Lunge (CSV)")
for f in ["Dataset/lunge/err.train.csv",
          "Dataset/lunge/err.test.csv",
          "Dataset/lunge/stage.train.csv",
          "Dataset/lunge/stage.test.csv",
          "Dataset/lunge/knee_angle.csv",
          "Dataset/lunge/knee_angle_2.csv"]:
    d, l = load_csv(f, "lunge")
    all_data += d; all_labels += l

# SQUAT (images)
print("\n[4] Squat (images)")
for folder, label in [
    ("Dataset/squat/train/Good",     "squat_correct"),
    ("Dataset/squat/train/Bad Back", "squat_wrong"),
    ("Dataset/squat/train/Bad Heel", "squat_wrong"),
    ("Dataset/squat/test/Good",      "squat_correct"),
    ("Dataset/squat/test/Bad Back",  "squat_wrong"),
    ("Dataset/squat/test/Bad Heel",  "squat_wrong"),
]:
    d, l = load_images(folder, label)
    all_data += d; all_labels += l

# ══════════════════════════════════════════════════════════════════════════════
print(f"\nTotal samples: {len(all_data)}")

if len(all_data) == 0:
    print("ERROR: No data loaded! Check your Dataset folder.")
    exit()

# Class distribution
unique, counts = np.unique(all_labels, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  {u:30s}: {c} samples")

# ── Split ─────────────────────────────────────────────────────────────────────
X = np.array(all_data)
y = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining : {len(X_train)} samples")
print(f"Testing  : {len(X_test)} samples")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (200 trees)...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/poseguard_model.pkl", "wb") as f:
    pickle.dump({'model': model, 'classes': list(model.classes_), 'accuracy': accuracy}, f)

print("\nModel saved → models/poseguard_model.pkl")
print(f"Exercises : {sorted(set([l.rsplit('_',1)[0] for l in all_labels]))}")
print("\nDone!")
