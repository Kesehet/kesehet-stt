import os
import sys
import time
import queue
import threading
import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from dotenv import load_dotenv
from pynput import keyboard
from faster_whisper import WhisperModel


# -----------------------------
# Discord
# -----------------------------
def post_to_discord(webhook_url: str, text: str):
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is missing. Put it in your .env file.")

    payload = {"content": text if text else "[No speech detected]"}
    r = requests.post(webhook_url, json=payload, timeout=15)
    r.raise_for_status()


# -----------------------------
# Audio Recording
# -----------------------------
class Recorder:
    def __init__(self, samplerate=16000, channels=1, device=None, dtype="float32"):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.dtype = dtype

        self._q = queue.Queue()
        self._frames = []
        self._stream = None

        self.is_recording = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # keep pushing chunks while recording
        self._q.put(indata.copy())

    def start(self):
        if self.is_recording.is_set():
            return

        self._frames = []
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.device,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        self.is_recording.set()

        # collector thread to pull from queue -> frames
        threading.Thread(target=self._collector_loop, daemon=True).start()

    def _collector_loop(self):
        while self.is_recording.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
                self._frames.append(chunk)
            except queue.Empty:
                pass

    def stop_and_save(self, outfile: str) -> str:
        if not self.is_recording.is_set():
            return ""

        self.is_recording.clear()
        time.sleep(0.2)  # small buffer to flush last chunks

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if not self._frames:
            return ""

        audio = np.concatenate(self._frames, axis=0)

        # mono
        if self.channels > 1:
            audio = np.mean(audio, axis=1)

        sf.write(outfile, audio, self.samplerate)
        return outfile


# -----------------------------
# Whisper Transcription
# -----------------------------
def transcribe_file(model: WhisperModel, wav_path: str, language: str | None, beam_size: int) -> str:
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    pieces = []
    for seg in segments:
        t = seg.text.strip()
        if t:
            pieces.append(t)

    text = " ".join(pieces).strip()

    # tiny UX print
    if info and getattr(info, "language", None):
        print(f"Language: {info.language} (prob={getattr(info, 'language_probability', 0):.2f})")

    return text


# -----------------------------
# Hotkey-driven CLI
# -----------------------------
def main():
    load_dotenv()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    parser = argparse.ArgumentParser(description="Press Q to start recording, Q again to stop+transcribe+send.")
    parser.add_argument("--model", default="tiny", help="tiny/base/small/medium/large-v3...")
    parser.add_argument("--compute_type", default="int8", help="int8 (best CPU), float16 (GPU), float32...")
    parser.add_argument("--language", default=None, help="en/hi/ur or leave empty for auto-detect")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", default=None, help="Input device index or name (optional)")
    parser.add_argument("--out", default="recording.wav", help="WAV filename to write")
    args = parser.parse_args()

    print(f"Loading faster-whisper model: {args.model} (cpu, compute_type={args.compute_type}) ...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    recorder = Recorder(
        samplerate=args.samplerate,
        channels=args.channels,
        device=args.device,
    )

    quit_flag = threading.Event()
    busy_flag = threading.Event()  # prevents spam keys during transcribe/post

    def print_help():
        print("\nControls:")
        print("  Q      → start/stop recording and transcribe+send to Discord")
        print("  CTRL+C → force quit\n")

    print_help()

    def on_press(key):
        # ignore if we're busy transcribing/posting
        if busy_flag.is_set():
            return

        try:
            # character keys
            if hasattr(key, "char") and key.char:
                c = key.char.lower()

                if c == "q":
                    if not recorder.is_recording.is_set():
                        # Start recording
                        recorder.start()
                        print("● Recording... (press Q again to stop) 💀‼‼ ")
                    else:
                        # Stop, transcribe, and send
                        busy_flag.set()
                        print("Stopping...")

                        wav = recorder.stop_and_save(args.out)
                        if not wav:
                            print("No audio captured.")
                            busy_flag.clear()
                            return

                        print("Transcribing...")
                        text = transcribe_file(model, wav, args.language, args.beam_size)

                        if not text:
                            print("No speech detected. Not sending.\n")
                        else:
                            print(f"\nTranscript:\n{text}\n")
                            print("Sending to Discord...")
                            try:
                                post_to_discord(webhook_url, text)
                                print("✅ Sent.\n")
                            except Exception as e:
                                print(f"❌ Discord send failed: {e}\n")

                        busy_flag.clear()
        except Exception as e:
            print(f"Key handler error: {e}", file=sys.stderr)

    # Start key listener
    with keyboard.Listener(on_press=on_press) as listener:
        while not quit_flag.is_set():
            time.sleep(0.1)

    print("Bye.")


if __name__ == "__main__":
    main()
