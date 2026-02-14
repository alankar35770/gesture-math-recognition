import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from collections import deque

# ============================================================
# LOAD STATIC MODEL
# ============================================================

static_model = pickle.load(open("gesture_knn_model.pkl", "rb"))
static_scaler = pickle.load(open("gesture_scaler.pkl", "rb"))
static_classes = json.load(open("class_names_knn.json"))

# ============================================================
# LOAD DYNAMIC MODEL
# ============================================================

dynamic_model = pickle.load(open("dynamic_gesture_model.pkl", "rb"))
dynamic_scaler = pickle.load(open("dynamic_gesture_scaler.pkl", "rb"))
dynamic_classes = json.load(open("dynamic_gesture_classes.json"))

# ============================================================
# MEDIAPIPE INIT
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ============================================================
# STATE VARIABLES
# ============================================================

gesture_buffer = deque(maxlen=5)

trajectory = []

mode = "gesture"      # gesture OR symbol
drawing = False

display_text = ""
confidence_text = ""
instruction_text = ""

MAX_POINTS = 150
MIN_POINTS = 30

# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_static(landmarks):

    pts = np.array(landmarks)

    pts = pts - pts[0]

    max_dist = np.max(np.linalg.norm(pts, axis=1))

    if max_dist > 0:
        pts = pts / max_dist

    return pts.flatten()


def normalize_dynamic(points):

    pts = np.array(points)

    pts = pts - pts[0]

    max_dist = np.max(np.linalg.norm(pts, axis=1))

    if max_dist > 0:
        pts = pts / max_dist

    return pts


def pad_dynamic(points):

    pts = list(points)

    if len(pts) > MAX_POINTS:
        pts = pts[:MAX_POINTS]

    while len(pts) < MAX_POINTS:
        pts.append(pts[-1])

    return np.array(pts).flatten()

# ============================================================
# CAMERA INIT
# ============================================================

cap = cv2.VideoCapture(0)

print("\nSYSTEM READY")
print("Press 1 → Gesture mode")
print("Press 2 → Symbol mode")
print("Press S → Start drawing")
print("Press E → Stop drawing")
print("Press Q → Quit\n")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # ========================================================
    # HAND DETECTION
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        # ====================================================
        # GESTURE MODE
        # ====================================================

        if mode == "gesture":

            norm = normalize_static(landmarks)

            norm = static_scaler.transform([norm])

            probs = static_model.predict_proba(norm)[0]

            pred = np.argmax(probs)

            gesture_buffer.append(pred)

            smooth_pred = max(set(gesture_buffer), key=gesture_buffer.count)

            gesture = static_classes[smooth_pred]

            confidence = probs[smooth_pred] * 100

            display_text = f"Gesture: {gesture}"
            confidence_text = f"Confidence: {confidence:.1f}%"
            instruction_text = "Press 2 for symbol mode"

        # ====================================================
        # SYMBOL MODE
        # ====================================================

        elif mode == "symbol":

            if drawing:

                fingertip = hand.landmark[8]

                trajectory.append([fingertip.x, fingertip.y])

                display_text = "Drawing integral..."
                confidence_text = ""
                instruction_text = "Press E to stop"

                # draw trajectory
                for i in range(1, len(trajectory)):

                    pt1 = (
                        int(trajectory[i-1][0]*640),
                        int(trajectory[i-1][1]*480)
                    )

                    pt2 = (
                        int(trajectory[i][0]*640),
                        int(trajectory[i][1]*480)
                    )

                    cv2.line(frame, pt1, pt2, (255,0,0), 2)

            else:

                display_text = "Symbol mode ready"
                confidence_text = ""
                instruction_text = "Press S to draw"

    # ========================================================
    # KEY HANDLING
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):

        mode = "gesture"
        drawing = False
        trajectory = []

        print("Gesture mode activated")

    elif key == ord('2'):

        mode = "symbol"
        drawing = False
        trajectory = []

        print("Symbol mode activated")

    elif key == ord('s') and mode == "symbol":

        drawing = True
        trajectory = []

        print("Drawing started")

    elif key == ord('e') and mode == "symbol" and drawing:

        drawing = False

        if len(trajectory) >= MIN_POINTS:

            traj = normalize_dynamic(trajectory)

            traj = pad_dynamic(traj)

            traj = dynamic_scaler.transform([traj])

            probs = dynamic_model.predict_proba(traj)[0]

            pred = np.argmax(probs)

            confidence = probs[pred] * 100

            symbol = dynamic_classes[pred]

            display_text = f"Symbol: {symbol}"
            confidence_text = f"Confidence: {confidence:.1f}%"
            instruction_text = ""

            print(f"Detected symbol: {symbol} ({confidence:.1f}%)")

        trajectory = []

    elif key == ord('q'):
        break

    # ========================================================
    # DISPLAY MENU
    # ========================================================

    cv2.putText(frame,
                f"MODE: {mode.upper()}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    cv2.putText(frame,
                display_text,
                (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(frame,
                confidence_text,
                (30,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2)

    cv2.putText(frame,
                instruction_text,
                (30,160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,200,0),
                2)

    cv2.imshow("Gesture + Integral Recognition System", frame)

cap.release()
cv2.destroyAllWindows()
