
# Gesture and Mathematical Symbol Recognition System 🖐️➗

A real-time computer vision system that recognizes hand gestures and dynamically drawn mathematical symbols using MediaPipe and Machine Learning. The system integrates static gesture recognition and trajectory-based symbol recognition with confidence estimation.

---

# 📌 Project Overview

This project provides an intuitive, touchless interface for interacting with computers using hand gestures and mathematical symbol drawing.

The system uses MediaPipe to extract 21 precise hand landmarks and applies Machine Learning models to classify:

- Static hand gestures
- Dynamically drawn mathematical symbols

Two independent recognition pipelines are integrated into one unified system.

---

# 🚀 Key Features

## Real-time Gesture Recognition

Recognizes gestures including:

- Left
- Right
- Up
- Down
- Palm
- Backward Palm
- Peace
- Yo

Features:

- Real-time prediction
- Confidence percentage display
- Landmark normalization (scale and translation invariant)
- Stable prediction using temporal smoothing
- Works with both left and right hand

---

## Mathematical Symbol Recognition

Recognizes dynamically drawn mathematical symbol:

- Integral (∫)

Features:

- Fingertip trajectory tracking using MediaPipe
- Real-time drawing visualization
- Trajectory normalization
- Machine learning based classification
- Confidence percentage output

---

## Dual Mode Operation

The system has two independent modes:

### Gesture Mode
Activated by pressing 1

-Detects static hand gestures with confidence.


### Symbol Mode
Activated by pressing 2

Allows drawing symbol using index finger.

Controls:

|Key| Function |
|---|----------|
| 1 | Activate gesture mode |
| 2 | Activate symbol mode |
| S | Start drawing symbol |
| E | Stop drawing and detect symbol |
| Q | Quit system |

---

# 🧠 System Architecture

The system consists of two independent machine learning pipelines.

---

## Static Gesture Recognition Pipeline

Input:

- 21 hand landmarks (x, y, z)

Processing:

- Translation normalization (relative to wrist)
- Scale normalization
- Feature scaling using StandardScaler

Model:

- K-Nearest Neighbors (KNN)

Output:

- Gesture label
- Confidence percentage

---

## Dynamic Symbol Recognition Pipeline

Input:

- Fingertip trajectory sequence

Processing:

- Trajectory normalization
- Fixed-length encoding
- Feature scaling using StandardScaler

Model:

- K-Nearest Neighbors (KNN)

Output:

- Symbol label
- Confidence percentage

---

# 🛠️ Tech Stack

Language:

- Python 3.11

Computer Vision:

- OpenCV
- MediaPipe

Machine Learning:

- Scikit-learn (KNN)

Data Processing:

- NumPy
- Pandas

Development Environment:

- VS Code

---

# 📂 Project Structure

gesture_project/
│
├── train_knn.py
├── test_knn.py
├── train_dynamic_gesture_model.py
│
├── static_landmarks_dataset_collector.py
├── dynamic_landmarks_dataset_collector.py
│
├── gesture_knn_model.pkl
├── gesture_scaler.pkl
├── class_names_knn.json
│
├── dynamic_gesture_model.pkl
├── dynamic_gesture_scaler.pkl
├── dynamic_gesture_classes.json
│
├── gesture_landmarks.csv
├── dynamic_landmarks_dataset.csv

# ▶️ How to Run

Run the system using:

python test_knn.py

# 👤 Author

Alankar Akinchan