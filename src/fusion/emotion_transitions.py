import sys, os, time, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.face_detector import FaceEmotionDetector

EMOTION_MAP = {"happiness":"happy","joy":"happy","sadness":"sad","anger":"angry","fearful":"fear","surprised":"surprise","calm":"neutral"}

def normalize_label(label):
    return EMOTION_MAP.get(label.lower(), label.lower())

def analyze_transitions(timeline):
    if len(timeline) < 2:
        return {"total_transitions": 0, "transitions": [], "stability": "N/A"}
    transitions = []
    for i in range(1, len(timeline)):
        prev = timeline[i-1]["emotion"]
        curr = timeline[i]["emotion"]
        if prev != curr:
            transitions.append({"from": prev, "to": curr, "at_second": timeline[i]["second"]})
    rate = len(transitions) / len(timeline)
    stability = "unstable" if rate > 0.3 else "moderate" if rate > 0.15 else "stable"
    return {"total_transitions": len(transitions), "transitions": transitions[-10:], "stability": stability}

def print_report(timeline, duration):
    r = analyze_transitions(timeline)
    print("\n" + "="*50)
    print("     EMOTION TRANSITION REPORT")
    print("="*50)
    print(f"Duration:          {duration} seconds")
    print(f"Total snapshots:   {len(timeline)}")
    print(f"Transitions:       {r['total_transitions']}")
    print(f"Stability:         {r['stability'].upper()}")
    if r["transitions"]:
        print("\nChanges detected:")
        for t in r["transitions"]:
            print(f"  [{t['at_second']:>3}s] {t['from']:<10} -> {t['to']}")
    else:
        print("\nNo changes - very stable.")
    print("="*50)

def run(duration_seconds=60):
    print(f"\n=== EMOTION TRANSITION TRACKER ===")
    print(f"Running for {duration_seconds}s. Press Q to stop.\n")
    face_model = FaceEmotionDetector()
    cap = cv2.VideoCapture(0)
    timeline = []
    start = time.time()
    end = start + duration_seconds
    last = None
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = int(time.time() - start)
        remaining = int(end - time.time())
        result = face_model.predict_from_frame(frame)
        if "top_emotion" in result:
            emotion = normalize_label(result["top_emotion"])
            conf = result["confidence"]
            timeline.append({"second": elapsed, "emotion": emotion, "confidence": conf})
            color = (0, 255, 0) if emotion == last else (0, 165, 255)
            last = emotion
            cv2.putText(frame, f"Left: {remaining//60:02d}:{remaining%60:02d}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"{emotion} ({conf*100:.0f}%)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Emotion Transitions - Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print_report(timeline, int(time.time() - start))

if __name__ == "__main__":
    mins = int(input("Track for how many minutes? "))
    run(mins * 60)
