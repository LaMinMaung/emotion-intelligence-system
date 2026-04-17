import sys
import os
import time
import cv2
import threading
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text.emotion_detector import EmotionDetector
from audio.audio_detector import AudioEmotionDetector
from vision.face_detector import FaceEmotionDetector

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

def normalize(scores: dict) -> dict:
    mapping = {
        "joy": "happy", "happiness": "happy",
        "sadness": "sad", "anger": "angry",
        "fear": "fear", "fearful": "fear",
        "surprise": "surprise", "surprised": "surprise",
        "disgust": "disgust", "neutral": "neutral", "calm": "neutral"
    }
    result = {e: 0.0 for e in EMOTIONS}
    for label, score in scores.items():
        mapped = mapping.get(label.lower())
        if mapped:
            result[mapped] += score
    return result


def get_duration():
    print("\n=== INTERVIEW MODE ===")
    print("How long is your session?")
    print("1. 5 minutes  (quick practice)")
    print("2. 15 minutes (short interview)")
    print("3. 30 minutes (full interview)")
    print("4. Custom duration")
    choice = input("\nEnter choice (1-4): ").strip()
    if choice == "1":
        return 5 * 60
    elif choice == "2":
        return 15 * 60
    elif choice == "3":
        return 30 * 60
    elif choice == "4":
        mins = int(input("Enter duration in minutes: "))
        return mins * 60
    else:
        return 5 * 60


def print_summary(emotion_log: list, duration_secs: int):
    print("\n" + "="*50)
    print("         SESSION SUMMARY")
    print("="*50)
    print(f"Total duration: {duration_secs // 60} min {duration_secs % 60} sec")
    print(f"Total samples:  {len(emotion_log)}\n")

    totals = defaultdict(float)
    for entry in emotion_log:
        for emotion, score in entry.items():
            totals[emotion] += score

    sorted_emotions = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    print("Average emotion scores across session:")
    for emotion, total in sorted_emotions:
        avg = total / len(emotion_log) if emotion_log else 0
        bar = "█" * int(avg * 30)
        print(f"  {emotion:<10} {bar} {avg*100:.1f}%")

    dominant = sorted_emotions[0][0] if sorted_emotions else "unknown"
    print(f"\nDominant emotion: {dominant.upper()}")

    # Feedback
    print("\n--- Interview Feedback ---")
    if dominant == "happy":
        print("Good job! You appeared confident and positive.")
    elif dominant == "neutral":
        print("You stayed calm and composed — good for interviews.")
    elif dominant in ["sad", "fear"]:
        print("You appeared nervous. Practice more to build confidence.")
    elif dominant == "angry":
        print("Try to relax — some tension was visible.")
    else:
        print("Mixed emotions detected — review your session.")
    print("="*50)


def run_interview():
    duration = get_duration()

    print("\nLoading models...")
    audio_model = AudioEmotionDetector()
    face_model = FaceEmotionDetector()
    print("Models ready!\n")

    emotion_log = []
    start_time = time.time()
    end_time = start_time + duration

    print(f"Session started! Running for {duration // 60} minutes.")
    print("Look at your camera. Press Q in the webcam window to stop early.\n")

    cap = cv2.VideoCapture(0)
    sample_count = 0

    while time.time() < end_time:
        elapsed = int(time.time() - start_time)
        remaining = int(end_time - time.time())
        mins_left = remaining // 60
        secs_left = remaining % 60

        # Capture face frame
        ret, frame = cap.read()
        if not ret:
            break

        face_result = face_model.predict_from_frame(frame)
        face_norm = normalize(face_result.get("all_scores", {}))

        # Show on webcam
        top = face_result.get("top_emotion", "")
        conf = face_result.get("confidence", 0)
        timer_text = f"Time left: {mins_left:02d}:{secs_left:02d}"
        emotion_text = f"{top} ({conf*100:.0f}%)" if top else ""

        cv2.putText(frame, timer_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, emotion_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Interview Mode - Press Q to stop", frame)

        if "all_scores" in face_result:
            emotion_log.append(face_norm)
            sample_count += 1

        # Every 30 seconds, record audio sample
        if sample_count % 30 == 1:
            print(f"[{elapsed//60:02d}:{elapsed%60:02d}] Recording 3s audio sample...")
            audio_result = audio_model.record_and_predict(duration=3)
            audio_norm = normalize(audio_result.get("all_scores", {}))
            emotion_log.append(audio_norm)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nSession stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    actual_duration = int(time.time() - start_time)
    print_summary(emotion_log, actual_duration)


if __name__ == "__main__":
    run_interview()