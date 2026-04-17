import sys, os, time, cv2
import sounddevice as sd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.face_detector import FaceEmotionDetector

STRESS_EMOTIONS = {"fear": 0.9, "angry": 0.8, "disgust": 0.6, "sad": 0.4, "surprise": 0.3, "neutral": 0.1, "happy": 0.0}
EMOTION_MAP = {"happiness":"happy","joy":"happy","sadness":"sad","anger":"angry","fearful":"fear","surprised":"surprise","calm":"neutral"}

def normalize_label(label):
    return EMOTION_MAP.get(label.lower(), label.lower())

def compute_stress_from_face(emotion_scores: dict) -> float:
    stress = 0.0
    for emotion, score in emotion_scores.items():
        label = normalize_label(emotion)
        weight = STRESS_EMOTIONS.get(label, 0.0)
        stress += weight * float(score)
    return min(round(stress, 3), 1.0)

def compute_stress_from_audio(duration=3) -> float:
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    rms = float(np.sqrt(np.mean(audio**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    stress = min((rms * 10) + (zcr * 2), 1.0)
    return round(stress, 3)

def stress_label(score: float) -> str:
    if score < 0.2: return "LOW"
    elif score < 0.4: return "MODERATE"
    elif score < 0.6: return "HIGH"
    else: return "VERY HIGH"

def run_stress_detector(duration_seconds=60):
    print(f"\n=== STRESS DETECTOR ===")
    print(f"Running for {duration_seconds}s. Press Q to stop.\n")
    face_model = FaceEmotionDetector()
    cap = cv2.VideoCapture(0)
    stress_log = []
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
        face_stress = 0.0
        if "all_scores" in result:
            face_stress = compute_stress_from_face(result["all_scores"])
        audio_stress = 0.0
        if sample_count % 60 == 0:
            print(f"[{elapsed}s] Sampling audio stress...")
            audio_stress = compute_stress_from_audio(duration=3)
        combined_stress = round(0.6 * face_stress + 0.4 * audio_stress, 3)
        stress_log.append(combined_stress)
        label = stress_label(combined_stress)
        color = (0,255,0) if combined_stress < 0.3 else (0,165,255) if combined_stress < 0.6 else (0,0,255)
        cv2.putText(frame, f"Left: {remaining//60:02d}:{remaining%60:02d}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Stress: {combined_stress*100:.0f}% ({label})", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Stress Detector - Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        sample_count += 1
    cap.release()
    cv2.destroyAllWindows()
    actual = int(time.time() - start)
    if stress_log:
        avg = round(sum(stress_log)/len(stress_log), 3)
        peak = round(max(stress_log), 3)
        print("\n" + "="*50)
        print("         STRESS REPORT")
        print("="*50)
        print(f"Duration:      {actual} seconds")
        print(f"Avg stress:    {avg*100:.1f}% ({stress_label(avg)})")
        print(f"Peak stress:   {peak*100:.1f}% ({stress_label(peak)})")
        print("\nFeedback:")
        if avg < 0.2:
            print("  Very calm. Excellent composure for an interview.")
        elif avg < 0.4:
            print("  Mild stress detected. Mostly composed — good.")
        elif avg < 0.6:
            print("  Moderate stress. Try deep breathing before interviews.")
        else:
            print("  High stress detected. Practice relaxation techniques.")
        print("="*50)

if __name__ == "__main__":
    mins = int(input("Run stress detection for how many minutes? "))
    run_stress_detector(mins * 60)
