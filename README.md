# Multimodal Emotion Intelligence System

A machine learning system that detects human emotions by combining three independent AI models — text, voice, and facial expression — into a single fused prediction.

## Demo

```
Type how you feel: feeling void
Text emotion:  sadness (97.1%)
Audio emotion: fearful (13.5%)
Face emotion:  sad    (95.7%)

FINAL FUSED EMOTION: SAD (confidence: 38.8%)
```

## Models Used

- **Text**: j-hartmann/emotion-english-distilroberta-base — RoBERTa fine-tuned on emotion classification
- **Audio**: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition — Wav2Vec2 trained on speech emotion
- **Vision**: DeepFace — CNN-based facial expression analysis

## Project Structure

```
src/
├── text/    # NLP emotion detection
├── audio/   # Speech emotion recognition
├── vision/  # Facial expression recognition
└── fusion/  # Multimodal weighted fusion
```

## Setup

```bash
git clone https://github.com/LaMinMaung/emotion-intelligence-system.git
cd emotion-intelligence-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tf-keras
```

## Run

```bash
python src/fusion/fusion_detector.py
```
## Applications

| Tool | Description |
|------|-------------|
| `fusion/fusion_detector.py` | Core fusion engine combining all 3 modalities |
| `fusion/stress_detector.py` | Real-time stress monitoring via face + voice |
| `fusion/confidence_scorer.py` | Interview confidence scoring with session report |
| `fusion/interview_mode.py` | Full interview session analyzer with feedback |
| `fusion/emotion_transitions.py` | Tracks how emotions shift over time |
| `fusion/interpreter.py` | Natural language interpretation of fused results |
| `audio/pitch_analyzer.py` | Pitch trend and energy level analysis from voice |
| `vision/gaze_detector.py` | Eye gaze direction detection via webcam |

## Research Context

Motivated by Poria et al. (2017) showing single-modality emotion recognition is insufficient for real-world use. This project implements late fusion across NLP, speech, and vision modalities.

## Author

La Min Maung — built as a technical portfolio project for graduate school applications.
