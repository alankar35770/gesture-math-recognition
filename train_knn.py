import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "gesture_landmarks.csv"

MODEL_FILE = "gesture_knn_model.pkl"

SCALER_FILE = "gesture_scaler.pkl"

CLASS_FILE = "class_names_knn.json"

K = 5

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_FILE)

labels = df.iloc[:, 0]

features = df.iloc[:, 1:].values

class_names = sorted(labels.unique())

print("Classes:", class_names)

# convert labels to numeric
label_map = {label: i for i, label in enumerate(class_names)}

y = np.array([label_map[label] for label in labels])

X = features

print("Samples:", len(X))

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# NORMALIZATION (VERY IMPORTANT)
# ============================================================

def normalize_landmarks(X):

    X_norm = []

    for row in X:

        pts = row.reshape(21,3)

        # translate to wrist origin
        pts = pts - pts[0]

        # scale normalize
        max_dist = np.max(np.linalg.norm(pts, axis=1))

        if max_dist > 0:
            pts = pts / max_dist

        X_norm.append(pts.flatten())

    return np.array(X_norm)

X_train = normalize_landmarks(X_train)

X_test = normalize_landmarks(X_test)

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining KNN model...")

model = KNeighborsClassifier(
    n_neighbors=K,
    weights="distance",
    metric="euclidean"
)

model.fit(X_train, y_train)

# ============================================================
# EVALUATE
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%\n")

print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================
# SAVE FILES
# ============================================================

pickle.dump(model, open(MODEL_FILE, "wb"))

pickle.dump(scaler, open(SCALER_FILE, "wb"))

json.dump(class_names, open(CLASS_FILE, "w"))

print("\nSaved:")
print(MODEL_FILE)
print(SCALER_FILE)
print(CLASS_FILE)
