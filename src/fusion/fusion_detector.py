import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text.emotion_detector import EmotionDetector
from audio.audio_detector import AudioEmotionDetector
from vision.face_detector import FaceEmotionDetector
import cv2

# Shared emotion labels across all three models
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

def normalize(scores: dict) -> dict:
    """Map any model's emotion labels to our shared set."""
    mapping = {
        "joy": "happy", "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "fear": "fear", "fearful": "fear",
        "surprise": "surprise", "surprised": "surprise",
        "disgust": "disgust",
        "neutral": "neutral", "calm": "neutral"
    }
    result = {e: 0.0 for e in EMOTIONS}
    for label, score in scores.items():
        mapped = mapping.get(label.lower(), None)
        if mapped:
            result[mapped] += score
    return result

def fuse(text_scores, audio_scores, face_scores) -> dict:
    """Weighted average fusion: text=40%, audio=20%, face=40%"""
    fused = {}
    for emotion in EMOTIONS:
        fused[emotion] = round(
            0.4 * text_scores.get(emotion, 0) +
            0.2 * audio_scores.get(emotion, 0) +
            0.4 * face_scores.get(emotion, 0),
            4
        )
    top = max(fused, key=fused.get)
    return {"top_emotion": top, "confidence": fused[top], "all_scores": fused}


if __name__ == "__main__":
    print("Loading all models...")
    text_model = EmotionDetector()
    audio_model = AudioEmotionDetector()
    face_model = FaceEmotionDetector()
    print("\nAll models ready!\n")

    # Step 1: get text input
    user_text = input("Type how you feel: ")
    text_result = text_model.predict(user_text)
    text_norm = normalize(text_result["all_scores"])
    print(f"Text emotion: {text_result['top_emotion']} ({text_result['confidence']*100:.1f}%)")

    # Step 2: record audio
    audio_result = audio_model.record_and_predict(duration=5)
    audio_norm = normalize(audio_result["all_scores"])
    print(f"Audio emotion: {audio_result['top_emotion']} ({audio_result['confidence']*100:.1f}%)")

    # Step 3: capture one face frame
    print("\nCapturing face... look at your camera!")
    cap = cv2.VideoCapture(0)
    face_scores_norm = {e: 0.0 for e in EMOTIONS}
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            result = face_model.predict_from_frame(frame)
            if "all_scores" in result:
                face_scores_norm = normalize(result["all_scores"])
                top = result["top_emotion"]
                conf = result["confidence"]
    cap.release()
    print(f"Face emotion: {top} ({conf*100:.1f}%)")

    # Step 4: fuse
    final = fuse(text_norm, audio_norm, face_scores_norm)
    print(f"\n{'='*40}")
    print(f"FINAL FUSED EMOTION: {final['top_emotion'].upper()}")
    print(f"Confidence: {final['confidence']*100:.1f}%")
    print(f"All scores: {final['all_scores']}")
    print(f"{'='*40}")