import cv2
import mediapipe as mp
import csv
import time
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "gesture_landmarks.csv"
CAPTURE_INTERVAL = 0.08

GESTURES = [
    "left",
    "right",
    "up",
    "down",
    "palm",
    "backward_palm",
    "peace",     # activates trajectory mode later
    "yo"         # activates gesture mode later
]

# key mapping
KEY_MAP = {
    ord('1'): "left",
    ord('2'): "right",
    ord('3'): "up",
    ord('4'): "down",
    ord('5'): "palm",
    ord('6'): "backward_palm",
    ord('7'): "peace",
    ord('8'): "yo"
}

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_csv():

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            header = ["label"]

            for i in range(21):

                header += [f"x{i}", f"y{i}", f"z{i}"]

            writer.writerow(header)


def initialize_mediapipe():

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    mp_draw = mp.solutions.drawing_utils

    return hands, mp_draw, mp_hands


def initialize_camera():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():

        print("ERROR: Camera not found")
        sys.exit()

    return cap


# ============================================================
# Menu
# ============================================================

def print_menu():

    print("\n================ LANDMARK DATASET COLLECTOR ================")
    print("\nSelect gesture:")
    print(" 1 → left")
    print(" 2 → right")
    print(" 3 → up")
    print(" 4 → down")
    print(" 5 → palm")
    print(" 6 → backward_palm")
    print(" 7 → peace   (trajectory mode trigger)")
    print(" 8 → yo      (gesture mode trigger)")
    print("\nControls:")
    print(" SPACE → Start/Stop recording")
    print(" Q     → Quit")
    print("============================================================\n")


def draw_ui(frame, active_label, recording, count):

    h, w, _ = frame.shape

    cv2.rectangle(frame, (0,0), (w,90), (30,30,30), -1)

    if active_label:

        text = f"Gesture: {active_label}"

        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,255), 2)

        text2 = f"Samples: {count[active_label]}"

        cv2.putText(frame, text2, (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,200,0), 2)

    if recording:

        cv2.putText(frame, "RECORDING",
                    (w-180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,255), 2)


# ============================================================
# DATA RECORDING
# ============================================================

def save_landmarks(label, landmarks):

    row = [label]

    for lm in landmarks:

        row.append(lm.x)
        row.append(lm.y)
        row.append(lm.z)

    with open(CSV_FILE, "a", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    initialize_csv()

    hands, mp_draw, mp_hands = initialize_mediapipe()

    cap = initialize_camera()

    print_menu()

    recording = False
    active_label = None
    last_capture = 0

    count = {g:0 for g in GESTURES}

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:

            hand_detected = True

            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if recording and active_label:

                now = time.time()

                if now - last_capture >= CAPTURE_INTERVAL:

                    save_landmarks(active_label,
                                   hand_landmarks.landmark)

                    count[active_label] += 1

                    print(f"{active_label}: {count[active_label]}")

                    last_capture = now

        draw_ui(frame, active_label, recording, count)

        if not hand_detected:

            cv2.putText(frame,
                        "No hand detected",
                        (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255),2)

        cv2.imshow("Static Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # gesture selection
        if key in KEY_MAP:

            active_label = KEY_MAP[key]

            print(f"Selected: {active_label}")

        # toggle recording
        elif key == 32:

            if active_label is None:

                print("Select gesture first")

            else:

                recording = not recording

                if recording:
                    print("Recording started")
                else:
                    print("Recording stopped")

        elif key == ord('q'):

            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
