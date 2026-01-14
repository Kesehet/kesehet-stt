import argparse
import sys
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


def record_audio(outfile: str, samplerate: int, channels: int, device=None, dtype="float32"):
    """
    Records audio until the user hits ENTER.
    Writes a WAV file to `outfile`.
    """
    print("\nPress ENTER to START recording...")
    input()
    print("Recording... press ENTER to STOP.\n")

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        device=device,
        dtype=dtype,
        callback=callback,
    )

    frames = []
    with stream:
        # Wait for Enter key to stop
        try:
            while True:
                if sys.stdin in select_inputs():
                    _ = sys.stdin.readline()
                    break
                frames.append(q.get())
        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt).")

    audio = np.concatenate(frames, axis=0)
    # Convert to mono if needed
    if channels > 1:
        audio = np.mean(audio, axis=1)

    sf.write(outfile, audio, samplerate)
    print(f"Saved recording to: {outfile}")
    return outfile


def select_inputs():
    """
    Cross-platform-ish non-blocking stdin check.
    Windows: uses msvcrt
    Others: uses select
    """
    if sys.platform.startswith("win"):
        import msvcrt
        if msvcrt.kbhit():
            # emulate "stdin is ready"
            return [sys.stdin]
        return []
    else:
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return r


def transcribe(model: WhisperModel, wav_path: str, language: str | None, beam_size: int):
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,          # helps skip long silences
        vad_parameters={"min_silence_duration_ms": 500},
    )

    print("\n--- Transcription ---")
    if info.language:
        print(f"Detected/Used language: {info.language} (prob={info.language_probability:.2f})")

    text_out = []
    for seg in segments:
        line = seg.text.strip()
        if line:
            text_out.append(line)

    final_text = " ".join(text_out).strip()
    print(final_text if final_text else "[No speech detected]")
    print("---------------------\n")
    return final_text


def main():
    parser = argparse.ArgumentParser(description="Record from mic and transcribe using faster-whisper.")
    parser.add_argument("--model", default="tiny", help="Model size: tiny, base, small, medium, large-v3, etc.")
    parser.add_argument("--device", default=None, help="Input device index or name (see sounddevice query).")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate (16000 is fine for Whisper).")
    parser.add_argument("--channels", type=int, default=1, help="1=mono (recommended).")
    parser.add_argument("--language", default=None, help="e.g. en, hi, ur (or leave empty for auto-detect).")
    parser.add_argument("--compute_type", default="int8", help="int8 (fast/light CPU), float16 (GPU), float32, etc.")
    parser.add_argument("--beam_size", type=int, default=3, help="Higher = more accurate, slower.")
    parser.add_argument("--out", default="recording.wav", help="Output wav filename.")
    args = parser.parse_args()

    # Load whisper model
    # device="cpu" keeps it lightweight; compute_type="int8" is usually the best for CPU speed/memory
    print(f"Loading model '{args.model}' (device=cpu, compute_type={args.compute_type}) ...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    # Record + transcribe in a loop
    print("\nTip: use --model tiny (fastest) or --model base for a bit more accuracy.")
    while True:
        try:
            record_audio(
                outfile=args.out,
                samplerate=args.samplerate,
                channels=args.channels,
                device=args.device,
            )
            transcribe(model, args.out, args.language, args.beam_size)

            print("Press ENTER to record again, or type q then ENTER to quit.")
            ans = input().strip().lower()
            if ans == "q":
                break
        except Exception as e:
            print(f"\nError: {e}\n")
            break


if __name__ == "__main__":
    main()
