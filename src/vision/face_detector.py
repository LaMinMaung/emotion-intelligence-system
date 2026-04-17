import cv2
from deepface import DeepFace

class FaceEmotionDetector:
    def __init__(self):
        print("Face emotion detector ready.")

    def predict_from_frame(self, frame) -> dict:
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emotions = result[0]["emotion"]
            scores = {k: round(v / 100, 4) for k, v in emotions.items()}
            top_emotion = max(scores, key=scores.get)
            return {
                "top_emotion": top_emotion,
                "confidence": scores[top_emotion],
                "all_scores": scores
            }
        except Exception as e:
            return {"error": str(e)}

    def run_live(self):
        print("Starting webcam... press Q to quit.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict_from_frame(frame)

            if "top_emotion" in result:
                label = f"{result['top_emotion']} ({result['confidence']*100:.1f}%)"
                cv2.putText(frame, label, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow("Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceEmotionDetector()
    detector.run_live()