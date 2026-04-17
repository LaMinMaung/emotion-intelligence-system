import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import pipeline

class AudioEmotionDetector:
    def __init__(self):
        print("Loading audio emotion model...")
        self.classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.sample_rate = 16000
        print("Audio model ready.")

    def record(self, duration: int = 5) -> np.ndarray:
        print(f"Recording for {duration} seconds... speak now!")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        print("Recording done.")
        return audio.flatten()

    def predict_from_array(self, audio: np.ndarray) -> dict:
        results = self.classifier(audio)
        scores = {r["label"]: round(r["score"], 4) for r in results}
        top_emotion = max(scores, key=scores.get)
        return {
            "top_emotion": top_emotion,
            "confidence": scores[top_emotion],
            "all_scores": scores
        }

    def predict_from_file(self, filepath: str) -> dict:
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        return self.predict_from_array(audio)

    def record_and_predict(self, duration: int = 5) -> dict:
        audio = self.record(duration)
        return self.predict_from_array(audio)


if __name__ == "__main__":
    detector = AudioEmotionDetector()
    print("\nRecording your voice...")
    result = detector.record_and_predict(duration=5)
    print(f"\nEmotion: {result['top_emotion']} ({result['confidence']*100:.1f}%)")
    print(f"All scores: {result['all_scores']}")