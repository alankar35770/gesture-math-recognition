import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIGURE
# ============================================================

CSV_FILE = "dynamic_landmarks_dataset.csv"

MODEL_FILE = "dynamic_gesture_model.pkl"

SCALER_FILE = "dynamic_gesture_scaler.pkl"

CLASS_FILE = "dynamic_gesture_classes.json"

K = 3

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

labels = df.iloc[:, 0]

features = df.iloc[:, 1:].values

class_names = sorted(labels.unique())

label_map = {label:i for i,label in enumerate(class_names)}

y = np.array([label_map[l] for l in labels])

X = features

print("Samples:", len(X))

# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# TRAIN
# ============================================================

print("Training model...")

model = KNeighborsClassifier(
    n_neighbors=K,
    weights='distance'
)

model.fit(X_train, y_train)

# ============================================================
# TEST
# ============================================================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy * 100, "%")

# ============================================================
# SAVE
# ============================================================

pickle.dump(model, open(MODEL_FILE, "wb"))

pickle.dump(scaler, open(SCALER_FILE, "wb"))

json.dump(class_names, open(CLASS_FILE, "w"))

print("Model saved.")
