import cv2
import mediapipe as mp
import numpy as np
import os
import time
from flask import Flask, render_template, request, Response, session
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# POSE DETECTOR
# =========================
class YogaPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Simple reference landmarks (demo purpose)
        self.target_poses = {
            "tree_pose": [(0.5, 0.3), (0.5, 0.5), (0.45, 0.7), (0.55, 0.7)],
            "warrior_pose": [(0.5, 0.35), (0.45, 0.55), (0.4, 0.75), (0.6, 0.75)],
            "downward_dog": [(0.5, 0.2), (0.4, 0.5), (0.6, 0.5), (0.5, 0.9)]
        }

    def calculate_accuracy(self, landmarks, target):
        score = 0
        count = 0

        for i, (tx, ty) in enumerate(target):
            lm = landmarks[i]
            if lm.visibility > 0.5:
                dist = np.sqrt((lm.x - tx)**2 + (lm.y - ty)**2)
                score += max(0, 1 - dist)
                count += 1

        return int((score / max(count, 1)) * 100)


# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.secret_key = "secret_key"

detector = YogaPoseDetector()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session["selected_pose"] = request.form.get("pose", "tree_pose")
        session["selected_time"] = request.form.get("time", "60")

    return render_template(
        "index.html",
        selected_pose=session.get("selected_pose", "tree_pose"),
        selected_time=session.get("selected_time", "60")
    )


def gen_frames(selected_pose, selected_time):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    end_time = start_time + int(selected_time)
    pose_completed = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(rgb)

        if results.pose_landmarks:
            detector.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                detector.mp_pose.POSE_CONNECTIONS
            )

            current_time = time.time()

            if current_time < end_time:
                accuracy = detector.calculate_accuracy(
                    results.pose_landmarks.landmark,
                    detector.target_poses[selected_pose]
                )

                cv2.putText(frame, "Pose Detection Started",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                cv2.putText(frame, f"Accuracy: {accuracy}%",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                if not pose_completed:
                    pose_completed = True

                cv2.putText(frame, "Pose Completed!",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(
            session.get("selected_pose", "tree_pose"),
            session.get("selected_time", "60")
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
