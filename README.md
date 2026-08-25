# ai-batch-two-topic-three

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.13 or higher
- pip
- pip install -r requirements.txt

## Setup

### Create a Virtual Environment

py -m venv venv

## Commands

### Audio File Input
	py audio_file_input.py
### Real time speech detection
	uvicorn audio_speech_detection:app --reload
	
The implementation uses **Silero VAD**, a lightweight speech model, to detect speech and silence boundaries for end-of-speech detection. 
**WhisperModel** is then used as an ASR speech model to transcribe the finalized speech segment.
