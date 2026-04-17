import librosa
import numpy as np
import sounddevice as sd


class PitchAnalyzer:
    def __init__(self):
        self.sample_rate = 16000

    def record(self, duration: int = 5) -> np.ndarray:
        print(f"Recording {duration}s... speak now!")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return audio.flatten()

    def analyze(self, audio: np.ndarray) -> dict:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 50:
                pitch_values.append(pitch)

        if not pitch_values:
            return {"error": "No pitch detected"}

        avg = float(np.mean(pitch_values))
        trend = float(pitch_values[-1] - pitch_values[0]) if len(pitch_values) > 1 else 0.0

        trend_label = "rising" if trend > 20 else "falling" if trend < -20 else "stable"
        energy_label = "high energy / excited" if avg > 200 else "normal" if avg > 120 else "low energy / calm"

        return {
            "avg_pitch_hz": round(avg, 2),
            "pitch_trend": trend_label,
            "energy_level": energy_label,
            "pitch_range_hz": round(float(max(pitch_values) - min(pitch_values)), 2),
            "samples": len(pitch_values)
        }

    def record_and_analyze(self, duration: int = 5) -> dict:
        audio = self.record(duration)
        return self.analyze(audio)


if __name__ == "__main__":
    analyzer = PitchAnalyzer()
    print("Speak naturally for 5 seconds...")
    result = analyzer.record_and_analyze(duration=5)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nAverage pitch:  {result['avg_pitch_hz']} Hz")
        print(f"Pitch trend:    {result['pitch_trend']}")
        print(f"Energy level:   {result['energy_level']}")
        print(f"Pitch range:    {result['pitch_range_hz']} Hz")