from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        print("Loading emotion model...")
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        print("Model ready.")

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"error": "Empty text provided"}

        results = self.classifier(text)

        if isinstance(results[0], list):
            scores = {item["label"]: round(item["score"], 4) for item in results[0]}
        else:
            scores = {results[0]["label"]: round(results[0]["score"], 4)}

        top_emotion = max(scores, key=scores.get)

        return {
            "text": text,
            "top_emotion": top_emotion,
            "confidence": scores[top_emotion],
            "all_scores": scores
        }


if __name__ == "__main__":
    detector = EmotionDetector()

    test_texts = [
        "I am so happy today!",
        "This makes me really angry.",
        "I feel scared and nervous.",
        "I am so sad and lonely."
    ]

    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {result['text']}")
        print(f"Emotion: {result['top_emotion']} ({result['confidence']*100:.1f}%)")