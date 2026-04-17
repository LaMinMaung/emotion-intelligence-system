import sys, os, time, cv2
import sounddevice as sd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.face_detector import FaceEmotionDetector

EMOTION_MAP = {"happiness":"happy","joy":"happy","sadness":"sad","anger":"angry","fearful":"fear","surprised":"surprise","calm":"neutral"}

def normalize_label(label):
    return EMOTION_MAP.get(label.lower(), label.lower())

def confidence_from_emotion(emotion_scores: dict) -> float:
    positive = {"happy": 1.0, "neutral": 0.6, "surprise": 0.3}
    negative = {"fear": -1.0, "sad": -0.8, "angry": -0.5, "disgust": -0.4}
    score = 0.0
    for emotion, val in emotion_scores.items():
        label = normalize_label(emotion)
        score += positive.get(label, 0.0) * float(val)
        score += negative.get(label, 0.0) * float(val)
    return min(max(round((score + 1) / 2, 3), 0.0), 1.0)

def confidence_from_voice(duration=3) -> float:
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    rms = float(np.sqrt(np.mean(audio**2)))
    volume_score = min(rms * 15, 1.0)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    stability = max(1.0 - zcr * 3, 0.0)
    return round((volume_score * 0.5 + stability * 0.5), 3)

def confidence_label(score: float) -> str:
    if score >= 0.75: return "HIGH"
    elif score >= 0.5: return "MODERATE"
    elif score >= 0.25: return "LOW"
    else: return "VERY LOW"

def run_confidence_scorer(duration_seconds=60):
    print(f"\n=== CONFIDENCE SCORER ===")
    print(f"Running for {duration_seconds}s. Press Q to stop.\n")
    face_model = FaceEmotionDetector()
    cap = cv2.VideoCapture(0)
    confidence_log = []
    start = time.time()
    end = start + duration_seconds
    sample_count = 0
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = int(time.time() - start)
        remaining = int(end - time.time())
        result = face_model.predict_from_frame(frame)
        face_conf = 0.5
        if "all_scores" in result:
            face_conf = confidence_from_emotion(result["all_scores"])
        voice_conf = 0.5
        if sample_count % 60 == 0:
            print(f"[{elapsed}s] Sampling voice confidence...")
            voice_conf = confidence_from_voice(duration=3)
        combined = round(0.6 * face_conf + 0.4 * voice_conf, 3)
        confidence_log.append(combined)
        label = confidence_label(combined)
        color = (0,255,0) if combined >= 0.6 else (0,165,255) if combined >= 0.35 else (0,0,255)
        cv2.putText(frame, f"Left: {remaining//60:02d}:{remaining%60:02d}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Confidence: {combined*100:.0f}% ({label})", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Confidence Scorer - Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        sample_count += 1
    cap.release()
    cv2.destroyAllWindows()
    actual = int(time.time() - start)
    if confidence_log:
        avg = round(sum(confidence_log)/len(confidence_log), 3)
        peak = round(max(confidence_log), 3)
        low = round(min(confidence_log), 3)
        print("\n" + "="*50)
        print("       CONFIDENCE REPORT")
        print("="*50)
        print(f"Duration:         {actual} seconds")
        print(f"Avg confidence:   {avg*100:.1f}% ({confidence_label(avg)})")
        print(f"Peak confidence:  {peak*100:.1f}% ({confidence_label(peak)})")
        print(f"Lowest point:     {low*100:.1f}% ({confidence_label(low)})")
        print("\nFeedback:")
        if avg >= 0.7:
            print("  Excellent confidence level. You come across as assured and credible.")
        elif avg >= 0.5:
            print("  Good confidence. Maintain eye contact and speak clearly.")
        elif avg >= 0.3:
            print("  Moderate confidence. Try smiling more and slowing your speech.")
        else:
            print("  Low confidence detected. Practice with mock interviews.")
        print("="*50)

if __name__ == "__main__":
    mins = int(input("Score confidence for how many minutes? "))
    run_confidence_scorer(mins * 60)
