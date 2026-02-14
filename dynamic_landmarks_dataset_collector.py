import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "dynamic_landmarks_dataset.csv"

LABEL = "integral"

MAX_POINTS = 150
MIN_POINTS = 30

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_csv():

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            header = ["label"]

            for i in range(MAX_POINTS):
                header += [f"x{i}", f"y{i}"]

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

        print("Camera not found")
        sys.exit()

    return cap


# ============================================================
# TRAJECTORY UTILITIES
# ============================================================

def normalize_trajectory(points):

    pts = np.array(points)

    pts = pts - pts[0]

    max_dist = np.max(np.linalg.norm(pts, axis=1))

    if max_dist > 0:
        pts = pts / max_dist

    return pts.tolist()


def pad_or_trim(points):

    if len(points) > MAX_POINTS:
        return points[:MAX_POINTS]

    while len(points) < MAX_POINTS:
        points.append(points[-1])

    return points


def save_trajectory(points):

    pts = normalize_trajectory(points)

    pts = pad_or_trim(pts)

    row = [LABEL]

    for p in pts:
        row.append(p[0])
        row.append(p[1])

    with open(CSV_FILE, "a", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_csv()

    hands, mp_draw, mp_hands = initialize_mediapipe()

    cap = initialize_camera()

    print("\nIntegral Dataset Collector")
    print("\nControls:")
    print(" SPACE → Start/Stop drawing")
    print(" S     → Save trajectory")
    print(" C     → Clear trajectory")
    print(" Q     → Quit\n")

    trajectory = []

    drawing = False

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        canvas = frame.copy()

        cv2.rectangle(canvas, (0,0), (640,80), (30,30,30), -1)

        cv2.putText(canvas,
                    "Symbol: integral",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,255),2)

        cv2.putText(canvas,
                    f"Samples: {count}",
                    (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,200,0),2)

        if drawing:

            cv2.putText(canvas,
                        "DRAWING",
                        (500,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,0,255),2)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingertip = hand_landmarks.landmark[8]

            x = int(fingertip.x * 640)
            y = int(fingertip.y * 480)

            cv2.circle(canvas, (x,y), 5, (0,255,0), -1)

            if drawing:

                trajectory.append([fingertip.x, fingertip.y])

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

            cv2.line(canvas, pt1, pt2, (255,0,0), 2)

        cv2.imshow("Integral Collector", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:

            drawing = not drawing

            if drawing:

                trajectory = []
                print("Drawing started")

            else:

                print("Drawing stopped")

        elif key == ord('c'):

            trajectory = []
            print("Cleared")

        elif key == ord('s'):

            if len(trajectory) >= MIN_POINTS:

                save_trajectory(trajectory)

                count += 1

                trajectory = []

                print(f"Saved sample {count}")

            else:

                print("Too few points")

        elif key == ord('q'):

            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
