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

## Research Context

Motivated by Poria et al. (2017) showing single-modality emotion recognition is insufficient for real-world use. This project implements late fusion across NLP, speech, and vision modalities.

## Author

La Min Maung — built as a technical portfolio project for graduate school applications.
