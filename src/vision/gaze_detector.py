import cv2
import numpy as np

class GazeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        print("Gaze detector ready.")

    def detect_gaze(self, frame) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {"gaze": "no_face", "looking_at_camera": False}

        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = self.eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            return {"gaze": "eyes_closed", "looking_at_camera": False}

        gaze_positions = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            _, threshold = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    rel_x = cx / ew
                    rel_y = cy / eh
                    gaze_positions.append((rel_x, rel_y))

        if not gaze_positions:
            return {"gaze": "unknown", "looking_at_camera": False}

        avg_x = np.mean([p[0] for p in gaze_positions])
        avg_y = np.mean([p[1] for p in gaze_positions])

        if 0.3 < avg_x < 0.7 and 0.3 < avg_y < 0.7:
            gaze = "center"
            looking = True
        elif avg_x < 0.3:
            gaze = "looking_left"
            looking = False
        elif avg_x > 0.7:
            gaze = "looking_right"
            looking = False
        else:
            gaze = "looking_away"
            looking = False

        return {
            "gaze": gaze,
            "looking_at_camera": looking,
            "gaze_x": round(float(avg_x), 3),
            "gaze_y": round(float(avg_y), 3)
        }

    def run_live(self):
        print("Starting gaze detection... press Q to quit.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.detect_gaze(frame)
            gaze = result.get("gaze", "unknown")
            looking = result.get("looking_at_camera", False)

            color = (0, 255, 0) if looking else (0, 0, 255)
            label = f"Gaze: {gaze}"
            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            status = "Looking at camera" if looking else "Looking away"
            cv2.putText(frame, status, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Gaze Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = GazeDetector()
    detector.run_live()