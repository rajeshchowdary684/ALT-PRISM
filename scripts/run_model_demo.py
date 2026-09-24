"""
PRISM Real-Time Terminal Live Microphone Voice Recorder & Speech Pipeline
===========================================================================
Records your voice live in real-time directly from your system microphone,
performs voice activity detection, and extracts text & English translation on the fly.
"""

import argparse
import queue
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sounddevice as sd

from ml.inference.engine import InferenceEngine
from ml.inference.language_detector import language_detector

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[MIC STREAM WARNING] {status}", file=sys.stderr)
    audio_queue.put(indata.copy().flatten())


def is_valid_speech_text(text: str) -> bool:
    if not text:
        return False
    cleaned = text.strip(" .?!,")
    if len(cleaned) < 2:
        return False
    lower = cleaned.lower()
    hallucinations = ["subtitles by", "amara.org", "thank you", "thanks for watching", "subscribe", "you"]
    if any(h in lower for h in hallucinations) and len(cleaned) < 20:
        return False
    return True


def run_realtime_voice_demo(language_hint: str = None):
    print("=" * 80)
    print("  PRISM REAL-TIME VOICE RECORDING & TRANSLATION ENGINE (TERMINAL)")
    print("=" * 80)
    print("[INIT] Loading local self-hosted AI models into memory...")

    engine = InferenceEngine()
    engine.initialize_models()

    sample_rate = 16000
    block_size = int(sample_rate * 0.4)  # 400ms audio chunks

    current_segment_chunks = []
    silence_count = 0
    segment_counter = 1
    total_audio_samples = 0
    full_transcripts = []
    full_translations = []

    print("\n" + "=" * 80)
    print("  [LIVE MIC ACTIVE] SPEAK INTO YOUR MICROPHONE NOW...")
    print(f"  (Language Hint: {language_hint.upper() if language_hint else 'AUTO-DETECT MULTILINGUAL'})")
    print("  Press Ctrl+C at any time to finish session & view full transcript summary.")
    print("=" * 80 + "\n")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=audio_callback,
        ):
            start_stream_time = time.time()

            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if len(chunk) == 0:
                    continue

                total_audio_samples += len(chunk)

                # Short-term RMS energy check for Voice Activity Detection
                frame_energy = float(np.sqrt(np.mean(np.square(chunk))))
                is_voice = frame_energy > 0.035  # Raised threshold to suppress background noise hallucination

                current_time = round(total_audio_samples / sample_rate, 2)

                if is_voice:
                    silence_count = 0
                    current_segment_chunks.append(chunk)
                    current_buffered = np.concatenate(current_segment_chunks)
                    segment_start = round((total_audio_samples - len(current_buffered)) / sample_rate, 2)

                    if len(current_buffered) >= sample_rate * 1.2:  # Need at least 1.2s of speech for partial
                        asr_res = engine.asr_provider.transcribe_segment(
                            current_buffered, sample_rate=sample_rate, language_hint=language_hint
                        )
                        if is_valid_speech_text(asr_res.text):
                            sys.stdout.write(
                                f"\r  [PARTIAL 00:{int(segment_start):02d} -> 00:{int(current_time):02d}] LISTEN: \"{asr_res.text}\"                "
                            )
                            sys.stdout.flush()
                else:
                    silence_count += 1
                    if silence_count >= 10:  # ~4 seconds of silence before finalizing segment
                        if len(current_segment_chunks) > 0:
                            segment_audio = np.concatenate(current_segment_chunks)
                            segment_start = round((total_audio_samples - len(segment_audio)) / sample_rate, 2)

                            if len(segment_audio) >= sample_rate * 1.0:  # Need at least 1 second of real speech
                                asr_res = engine.asr_provider.transcribe_segment(
                                    segment_audio, sample_rate=sample_rate, language_hint=language_hint
                                )
                                if is_valid_speech_text(asr_res.text):
                                    cs_res = engine.code_switch_analyzer.analyze(
                                        asr_res.text, language_hint=asr_res.language
                                    )
                                    lang_det = language_detector.detect_from_text(
                                        asr_res.text, language_hint=cs_res.primary_language
                                    )
                                    trans_res = engine.translation_provider.translate(
                                        asr_res.text,
                                        source_lang=lang_det.language_code,
                                        target_lang="en",
                                    )

                                    if trans_res.translated_text:
                                        full_transcripts.append(asr_res.text)
                                        full_translations.append(trans_res.translated_text)

                                        sys.stdout.write("\r" + " " * 90 + "\r")  # Clear partial line
                                        print(f"  [SEGMENT #{segment_counter:02d} | 00:{int(segment_start):02d} -> 00:{int(current_time):02d}]")
                                        print(f"  |-- ORIGINAL ({lang_det.language_name.upper()}): \"{asr_res.text}\"")
                                        print(f"  +-- ENGLISH TRANSLATION : \"{trans_res.translated_text}\"")
                                        if cs_res.is_code_switched:
                                            print(f"     [CODE-SWITCHED: {', '.join(cs_res.languages).upper()}]")
                                        print("-" * 80)

                                        segment_counter += 1

                            current_segment_chunks = []
                        silence_count = 0

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("          REAL-TIME VOICE SESSION COMPLETED")
        print("=" * 80)

        final_orig = " ".join(full_transcripts).strip()
        final_eng = " ".join(full_translations).strip()

        print(f"  TOTAL RECORDING TIME  : {total_audio_samples / sample_rate:.2f} seconds")
        print(f"  TOTAL SEGMENTS        : {len(full_transcripts)}")
        print("-" * 80)
        print(f"  FULL ORIGINAL TRANSCRIPT :\n  -> \"{final_orig if final_orig else 'No speech detected.'}\"")
        print("-" * 80)
        print(f"  FULL ENGLISH TRANSLATION :\n  -> \"{final_eng if final_eng else 'No translation generated.'}\"")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PRISM Real-time Terminal Voice Recorder & Speech Pipeline")
    parser.add_argument("--lang", type=str, default=None, help="Optional language hint (te, hi, ta, kn, mr, bn, ml, en)")
    args = parser.parse_args()

    run_realtime_voice_demo(language_hint=args.lang)


if __name__ == "__main__":
    main()