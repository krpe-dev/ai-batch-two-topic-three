import json
import time
import os
import asyncio
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# This backend expects frontend to send 16kHz PCM int16 audio.
SAMPLE_RATE = 16000
FRAME_SIZE = 512

ASR_MODEL_SIZE = "small"

# Easier speech start detection
VAD_THRESHOLD = 0.40
MIN_RMS_FOR_SPEECH = 0.003
MIN_SPEECH_FRAMES = 2

# While user is answering
CAPTURE_VAD_THRESHOLD = 0.45

# Capture duration
MIN_CAPTURE_SECONDS = 0.8
MAX_CAPTURE_SECONDS = 120.0

# End after real silent audio frames
END_SILENCE_SECONDS = 1.2
END_SILENCE_FRAMES = int(END_SILENCE_SECONDS * SAMPLE_RATE / FRAME_SIZE)

# Ignore TTS tail after AI finishes speaking
IGNORE_AUDIO_AFTER_AI_SECONDS = 0.5

# Delay before next question
ASK_NEXT_QUESTION_DELAY_SECONDS = 0.2

# Reject fake/too-short answers
MIN_VOICE_SECONDS_TO_ACCEPT = 0.35
MIN_WORDS_TO_ACCEPT = 2

# Pre-roll prevents missing first words
PRE_ROLL_SECONDS = 0.6
PRE_ROLL_SAMPLES = int(PRE_ROLL_SECONDS * SAMPLE_RATE)

DEBUG = True


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "What is polymorphism in object-oriented programming?",
    "Can you explain inheritance with an example?",
    "What is the difference between a list and a tuple in Python?",
    "Why should we handle exceptions in code?"
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Whisper model...")
asr_model = WhisperModel(
    ASR_MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)
print("Whisper loaded.")

print("Loading Silero VAD...")
vad_model = load_silero_vad()
print("Silero VAD loaded.")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ASRResult:
    transcript: str
    latest_end_time: float
    avg_logprob: float = -2.0
    no_speech_prob: float = 1.0
    compression_ratio: float = 0.0


# ============================================================
# HELPERS
# ============================================================

def log(*args):
    if DEBUG:
        print(*args)


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"00:{minutes:02d}:{secs:04.1f}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def rms_energy(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame)) + 1e-8))


def normalize_text(text: str) -> str:
    text = text.lower().strip()

    for ch in [".", ",", "!", "?", "\"", "'"]:
        text = text.replace(ch, "")

    return " ".join(text.split())


def is_likely_whisper_hallucination(text: str) -> bool:
    normalized = normalize_text(text)

    hallucination_phrases = {
        "thank you for watching",
        "thanks for watching",
        "thats the end of the video",
        "that is the end of the video",
        "thank you thank you so much",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "im not saying that"
    }

    if normalized in hallucination_phrases:
        return True

    for phrase in hallucination_phrases:
        if phrase in normalized:
            return True

    return False


def calculate_dynamic_confidence(
    final_reason: str,
    final_asr: ASRResult,
    transcript: str,
    captured_seconds: float,
    capture_speech_frames: int,
    capture_total_frames: int
) -> float:
    word_count = len(transcript.split())

    logprob_score = (final_asr.avg_logprob + 1.4) / 1.4
    logprob_score = clamp(logprob_score, 0.0, 1.0)

    no_speech_score = 1.0 - clamp(final_asr.no_speech_prob, 0.0, 1.0)

    if final_asr.compression_ratio <= 0:
        compression_score = 0.80
    elif final_asr.compression_ratio <= 2.4:
        compression_score = 1.0
    elif final_asr.compression_ratio <= 3.0:
        compression_score = 0.75
    else:
        compression_score = 0.55

    asr_score = (
        0.55 * logprob_score +
        0.35 * no_speech_score +
        0.10 * compression_score
    )

    if capture_total_frames > 0:
        speech_ratio = capture_speech_frames / capture_total_frames
    else:
        speech_ratio = 0.0

    if 0.20 <= speech_ratio <= 0.90:
        vad_score = 0.92
    elif 0.10 <= speech_ratio < 0.20:
        vad_score = 0.78
    else:
        vad_score = 0.65

    if final_reason == "no_voice":
        eos_score = 0.92
    elif final_reason == "max_capture":
        eos_score = 0.78
    else:
        eos_score = 0.82

    if word_count >= 12:
        length_score = 0.95
    elif word_count >= 6:
        length_score = 0.88
    elif word_count >= 3:
        length_score = 0.78
    elif word_count >= 1:
        length_score = 0.65
    else:
        length_score = 0.45

    if captured_seconds >= 3.0:
        duration_score = 0.92
    elif captured_seconds >= 1.5:
        duration_score = 0.82
    else:
        duration_score = 0.65

    content_score = 0.65 * length_score + 0.35 * duration_score

    confidence = (
        0.35 * asr_score +
        0.30 * vad_score +
        0.25 * eos_score +
        0.10 * content_score
    )

    confidence = clamp(confidence, 0.50, 0.95)

    return round(confidence, 2)


# ============================================================
# ASR
# ============================================================

def transcribe_audio(audio: np.ndarray) -> ASRResult:
    if audio is None or len(audio) == 0:
        return ASRResult(
            transcript="",
            latest_end_time=0.0,
            avg_logprob=-2.0,
            no_speech_prob=1.0,
            compression_ratio=0.0
        )

    segments, _ = asr_model.transcribe(
        audio.astype(np.float32),
        language="en",
        beam_size=1,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0
    )

    parts = []
    latest_end = 0.0

    logprobs = []
    no_speech_probs = []
    compression_ratios = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            parts.append(text)

        latest_end = max(latest_end, float(segment.end))

        if hasattr(segment, "avg_logprob"):
            logprobs.append(float(segment.avg_logprob))

        if hasattr(segment, "no_speech_prob"):
            no_speech_probs.append(float(segment.no_speech_prob))

        if hasattr(segment, "compression_ratio"):
            compression_ratios.append(float(segment.compression_ratio))

    transcript = " ".join(parts).strip()

    avg_logprob = float(np.mean(logprobs)) if logprobs else -2.0
    no_speech_prob = float(np.mean(no_speech_probs)) if no_speech_probs else 1.0
    compression_ratio = float(np.mean(compression_ratios)) if compression_ratios else 0.0

    return ASRResult(
        transcript=transcript,
        latest_end_time=latest_end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


@app.get("/")
async def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "static", "speech-detection.html")

    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected from browser.")

    state = "IDLE"

    question_index = 0
    answers = []
    interview_finished = False

    incoming_audio_buffer = np.zeros(0, dtype=np.float32)
    capture_audio_buffer = np.zeros(0, dtype=np.float32)
    pre_roll_buffer = np.zeros(0, dtype=np.float32)

    total_samples = 0
    capture_start_sample = 0

    speech_frames = 0
    capturing = False
    consecutive_silence_frames = 0

    capture_total_frames = 0
    capture_speech_frames = 0

    ignore_audio_until_time = 0.0
    audio_debug_counter = 0

    # ============================================================
    # SAFE SEND
    # ============================================================

    async def safe_send_json(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except WebSocketDisconnect:
            print("WebSocket disconnected while sending.")
            return False
        except RuntimeError as e:
            print("WebSocket already closed while sending:", repr(e))
            return False
        except Exception as e:
            print("Failed to send websocket message:", repr(e))
            return False

    # ============================================================
    # INNER HELPERS
    # ============================================================

    async def send_question():
        nonlocal state

        if question_index >= len(QUESTIONS):
            return False

        ok = await safe_send_json({
            "event": "ai_response",
            "text": QUESTIONS[question_index],
            "question_number": question_index + 1,
            "total_questions": len(QUESTIONS)
        })

        if not ok:
            return False

        print(f"Asked question {question_index + 1}: {QUESTIONS[question_index]}")

        state = "AI_SPEAKING"
        return True

    async def reset_capture_state():
        nonlocal capture_audio_buffer
        nonlocal pre_roll_buffer
        nonlocal speech_frames
        nonlocal capturing
        nonlocal consecutive_silence_frames
        nonlocal capture_total_frames
        nonlocal capture_speech_frames

        capture_audio_buffer = np.zeros(0, dtype=np.float32)
        pre_roll_buffer = np.zeros(0, dtype=np.float32)

        speech_frames = 0
        capturing = False
        consecutive_silence_frames = 0

        capture_total_frames = 0
        capture_speech_frames = 0

    async def finish_current_answer(
        user_text: str,
        final_asr: ASRResult,
        final_reason: str,
        captured_seconds: float
    ):
        nonlocal question_index
        nonlocal interview_finished
        nonlocal state

        relative_end = final_asr.latest_end_time

        if relative_end <= 0:
            relative_end = captured_seconds

        absolute_end = (capture_start_sample / SAMPLE_RATE) + relative_end
        end_of_speech = format_timestamp(absolute_end)

        confidence = calculate_dynamic_confidence(
            final_reason=final_reason,
            final_asr=final_asr,
            transcript=user_text,
            captured_seconds=captured_seconds,
            capture_speech_frames=capture_speech_frames,
            capture_total_frames=capture_total_frames
        )

        current_question = QUESTIONS[question_index]

        answer_record = {
            "question_number": question_index + 1,
            "question": current_question,
            "answer": user_text,
            "end_of_speech": end_of_speech,
            "confidence": confidence,
            "final_reason": final_reason,
            "captured_seconds": round(captured_seconds, 2)
        }

        ok = await safe_send_json({
            "status": "success",
            "end_of_speech": end_of_speech,
            "confidence": confidence
        })

        if not ok:
            return

        answers.append(answer_record)

        print("Saved answer:")
        print(json.dumps(answer_record, indent=2))

        ok = await safe_send_json({
            "event": "answer_saved",
            **answer_record
        })

        if not ok:
            return

        question_index += 1

        await reset_capture_state()

        if question_index < len(QUESTIONS):
            await asyncio.sleep(ASK_NEXT_QUESTION_DELAY_SECONDS)
            await send_question()
        else:
            interview_finished = True
            state = "IDLE"

            final_output = {
                "event": "final_output",
                "status": "completed",
                "total_questions": len(QUESTIONS),
                "questions": answers
            }

            await safe_send_json(final_output)

            print("Interview complete.")
            print("Final output:")
            print(json.dumps(final_output, indent=2))

    # Ask first question
    await send_question()

    try:
        while True:
            message = await websocket.receive()

            # ====================================================
            # TEXT CONTROL MESSAGE FROM BROWSER
            # ====================================================

            text_message = message.get("text")

            if text_message is not None:
                try:
                    command = json.loads(text_message)
                except Exception:
                    continue

                command_type = command.get("type")

                if command_type == "state":
                    old_state = state
                    state = command.get("state", "IDLE")

                    print("Browser state:", state)

                    if old_state == "AI_SPEAKING" and state == "IDLE":
                        incoming_audio_buffer = np.zeros(0, dtype=np.float32)

                        await reset_capture_state()

                        ignore_audio_until_time = (
                            time.time() + IGNORE_AUDIO_AFTER_AI_SECONDS
                        )

                        print(
                            f"Ignoring mic audio for {IGNORE_AUDIO_AFTER_AI_SECONDS}s after AI speech."
                        )

                elif command_type == "reset":
                    incoming_audio_buffer = np.zeros(0, dtype=np.float32)
                    capture_audio_buffer = np.zeros(0, dtype=np.float32)
                    pre_roll_buffer = np.zeros(0, dtype=np.float32)

                    total_samples = 0
                    capture_start_sample = 0

                    speech_frames = 0
                    capturing = False
                    consecutive_silence_frames = 0

                    capture_total_frames = 0
                    capture_speech_frames = 0

                    ignore_audio_until_time = 0.0

                    question_index = 0
                    answers = []
                    interview_finished = False
                    state = "IDLE"

                    await safe_send_json({
                        "event": "reset_done"
                    })

                    await send_question()

                continue

            if interview_finished:
                continue

            # ====================================================
            # AUDIO MESSAGE
            # ====================================================

            pcm_bytes = message.get("bytes")

            if pcm_bytes is None:
                continue

            int16_audio = np.frombuffer(pcm_bytes, dtype=np.int16)

            if len(int16_audio) == 0:
                continue

            # IMPORTANT:
            # No backend resampling here.
            # Frontend must send 16kHz PCM int16.
            float_audio = int16_audio.astype(np.float32) / 32768.0

            chunk_rms = float(np.sqrt(np.mean(float_audio ** 2) + 1e-8))
            chunk_max = float(np.max(np.abs(float_audio)))

            audio_debug_counter += 1

            if audio_debug_counter % 20 == 0:
                print(
                    "Audio chunk:",
                    "samples=", len(float_audio),
                    "rms=", round(chunk_rms, 4),
                    "max=", round(chunk_max, 4)
                )

                # await safe_send_json({
                #     "event": "audio_debug",
                #     "samples": int(len(float_audio)),
                #     "rms": round(chunk_rms, 4),
                #     "max": round(chunk_max, 4)
                # })

            incoming_audio_buffer = np.concatenate([
                incoming_audio_buffer,
                float_audio
            ])

            # ====================================================
            # PROCESS EXACT 512-SAMPLE 16k FRAMES
            # ====================================================

            while len(incoming_audio_buffer) >= FRAME_SIZE:
                frame = incoming_audio_buffer[:FRAME_SIZE]
                incoming_audio_buffer = incoming_audio_buffer[FRAME_SIZE:]

                total_samples += FRAME_SIZE

                # Ignore mic while AI is speaking
                if state == "AI_SPEAKING":
                    continue

                # Ignore TTS tail/noise after AI finishes speaking
                if time.time() < ignore_audio_until_time:
                    speech_frames = 0
                    capturing = False
                    consecutive_silence_frames = 0
                    capture_audio_buffer = np.zeros(0, dtype=np.float32)
                    pre_roll_buffer = np.zeros(0, dtype=np.float32)
                    continue

                # Maintain pre-roll
                pre_roll_buffer = np.concatenate([
                    pre_roll_buffer,
                    frame
                ])

                if len(pre_roll_buffer) > PRE_ROLL_SAMPLES:
                    pre_roll_buffer = pre_roll_buffer[-PRE_ROLL_SAMPLES:]

                with torch.no_grad():
                    vad_prob = vad_model(
                        torch.from_numpy(frame).float(),
                        SAMPLE_RATE
                    ).item()

                energy = rms_energy(frame)

                # ====================================================
                # SPEECH DETECTION
                # ====================================================

                if capturing:
                    is_speech = vad_prob >= CAPTURE_VAD_THRESHOLD
                else:
                    is_speech = (
                        vad_prob >= VAD_THRESHOLD
                        and energy >= MIN_RMS_FOR_SPEECH
                    )

                # ====================================================
                # SPEECH / SILENCE TRACKING
                # ====================================================

                if is_speech:
                    speech_frames += 1

                    if capturing:
                        consecutive_silence_frames = 0
                else:
                    speech_frames = 0

                    if capturing:
                        consecutive_silence_frames += 1

                # ====================================================
                # USER STARTS SPEAKING
                # ====================================================

                if state == "IDLE" and speech_frames >= MIN_SPEECH_FRAMES:
                    state = "LISTENING"
                    capturing = True
                    consecutive_silence_frames = 0

                    capture_start_sample = max(
                        0,
                        total_samples - len(pre_roll_buffer)
                    )

                    capture_audio_buffer = pre_roll_buffer.copy()

                    capture_total_frames = max(
                        1,
                        len(capture_audio_buffer) // FRAME_SIZE
                    )

                    capture_speech_frames = MIN_SPEECH_FRAMES

                    await safe_send_json({
                        "event": "speech_started",
                        "vad_probability": round(float(vad_prob), 3),
                        "rms": round(float(energy), 4)
                    })

                    print(
                        "Speech started.",
                        "vad=", round(float(vad_prob), 3),
                        "rms=", round(float(energy), 4)
                    )

                    continue

                if not capturing:
                    # Send VAD debug sometimes before speech starts
                    if total_samples % (FRAME_SIZE * 30) == 0:
                        await safe_send_json({
                            "event": "vad_idle_debug",
                            "vad_probability": round(float(vad_prob), 3),
                            "rms": round(float(energy), 4),
                            "speech_frames": speech_frames
                        })

                    continue

                # ====================================================
                # CAPTURE AUDIO
                # ====================================================

                capture_audio_buffer = np.concatenate([
                    capture_audio_buffer,
                    frame
                ])

                capture_total_frames += 1

                if is_speech:
                    capture_speech_frames += 1

                captured_seconds = len(capture_audio_buffer) / SAMPLE_RATE
                voice_seconds = capture_speech_frames * FRAME_SIZE / SAMPLE_RATE

                # Debug every ~0.5 seconds
                if capture_total_frames % 15 == 0:
                    # await safe_send_json({
                    #     "event": "vad_debug",
                    #     "vad_probability": round(float(vad_prob), 3),
                    #     "rms": round(float(energy), 4),
                    #     "is_speech": bool(is_speech),
                    #     "speech_frames": speech_frames,
                    #     "silence_frames": consecutive_silence_frames,
                    #     "needed_silence_frames": END_SILENCE_FRAMES,
                    #     "captured_seconds": round(captured_seconds, 2),
                    #     "voice_seconds": round(voice_seconds, 2)
                    # })

                    print(
                        "VAD debug:",
                        "vad=", round(float(vad_prob), 3),
                        "rms=", round(float(energy), 4),
                        "is_speech=", is_speech,
                        "silence_frames=", consecutive_silence_frames,
                        "needed=", END_SILENCE_FRAMES
                    )

                # ====================================================
                # END CONDITIONS
                # ====================================================

                no_voice_ready = (
                    captured_seconds >= MIN_CAPTURE_SECONDS
                    and voice_seconds >= MIN_VOICE_SECONDS_TO_ACCEPT
                    and consecutive_silence_frames >= END_SILENCE_FRAMES
                )

                max_capture_ready = captured_seconds >= MAX_CAPTURE_SECONDS

                if not no_voice_ready and not max_capture_ready:
                    continue

                if no_voice_ready:
                    final_reason = "no_voice"
                    print("Finalizing: silence detected.")
                else:
                    final_reason = "max_capture"
                    print("Finalizing: max capture reached.")

                # ====================================================
                # FINAL TRANSCRIPTION
                # ====================================================

                final_asr = await asyncio.to_thread(
                    transcribe_audio,
                    capture_audio_buffer
                )

                user_text = final_asr.transcript.strip()

                word_count = len(user_text.split())
                voice_seconds = capture_speech_frames * FRAME_SIZE / SAMPLE_RATE

                should_reject = (
                    word_count < MIN_WORDS_TO_ACCEPT
                    or voice_seconds < MIN_VOICE_SECONDS_TO_ACCEPT
                    or is_likely_whisper_hallucination(user_text)
                )

                if should_reject:
                    print("Rejected false/too-short/hallucinated capture.")
                    print("Transcript:", user_text)
                    print("Words:", word_count)
                    print("Voice seconds:", round(voice_seconds, 2))

                    await reset_capture_state()

                    await safe_send_json({
                        "event": "retry_question",
                        "message": "I did not hear a clear answer. Please answer again."
                    })

                    await asyncio.sleep(0.3)
                    await send_question()

                    break

                await finish_current_answer(
                    user_text=user_text,
                    final_asr=final_asr,
                    final_reason=final_reason,
                    captured_seconds=captured_seconds
                )

                break

    except WebSocketDisconnect:
        print("Browser disconnected.")

    except Exception as e:
        print("WebSocket backend error:", repr(e))
        traceback.print_exc()

        try:
            await safe_send_json({
                "event": "server_error",
                "message": str(e)
            })
        except Exception:
            pass