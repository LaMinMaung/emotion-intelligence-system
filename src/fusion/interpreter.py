class EmotionInterpreter:
    def interpret(self, text_result: dict, audio_result: dict,
                  face_result: dict, fused_result: dict) -> str:

        top = fused_result.get("top_emotion", "unknown")
        conf = fused_result.get("confidence", 0) * 100

        text_emotion = text_result.get("top_emotion", "unknown")
        audio_emotion = audio_result.get("top_emotion", "unknown")
        face_emotion = face_result.get("top_emotion", "unknown")

        lines = []
        lines.append(f"Overall emotional state: {top.upper()} ({conf:.1f}% confidence)")
        lines.append("")
        lines.append("Breakdown:")
        lines.append(f"  - What you said suggests: {text_emotion}")
        lines.append(f"  - Your voice tone suggests: {audio_emotion}")
        lines.append(f"  - Your facial expression suggests: {face_emotion}")
        lines.append("")

        if text_emotion == audio_emotion == face_emotion:
            lines.append("All three modalities agree — high confidence in this reading.")
        elif text_emotion != face_emotion:
            lines.append("Note: Your words and facial expression differ — possible masking.")
        elif audio_emotion != face_emotion:
            lines.append("Note: Voice tone and face differ — possible mixed emotions.")

        lines.append("")
        lines.append(self._advice(top))
        return "\n".join(lines)

    def _advice(self, emotion: str) -> str:
        advice = {
            "happy": "You appear positive and engaged. Great state for communication.",
            "sad": "You seem low. Take a moment — speaking to someone may help.",
            "angry": "Elevated tension detected. Consider pausing before responding.",
            "fear": "Anxiety detected. Slow breathing may help calm your nervous system.",
            "surprise": "High alertness detected. You appear engaged and reactive.",
            "disgust": "Discomfort detected. It may help to identify the source.",
            "neutral": "You appear calm and composed. Good baseline state."
        }
        return advice.get(emotion, "Emotional state noted.")


if __name__ == "__main__":
    interpreter = EmotionInterpreter()

    sample_text = {"top_emotion": "joy", "confidence": 0.96}
    sample_audio = {"top_emotion": "happy", "confidence": 0.72}
    sample_face = {"top_emotion": "happy", "confidence": 0.88}
    sample_fused = {"top_emotion": "happy", "confidence": 0.89,
                    "all_scores": {"happy": 0.89, "neutral": 0.06}}

    output = interpreter.interpret(sample_text, sample_audio,
                                   sample_face, sample_fused)
    print(output)