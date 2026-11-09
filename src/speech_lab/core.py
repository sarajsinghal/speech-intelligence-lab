"""Audio, transcript scoring, validation, and export helpers."""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import wave
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEMO_REFERENCE = "आज we are testing speech recognition in Hinglish"
DEMO_HYPOTHESIS = "आज we are test speech recognition in Hinglish"


class AudioValidationError(ValueError):
    """Raised for unsupported or malformed audio uploads."""


def validate_audio(data: bytes, filename: str, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(f"Unsupported file type: {suffix or '(no extension)'}. Use WAV, MP3, M4A, FLAC or OGG.")
    if not data:
        raise AudioValidationError("The audio file is empty.")
    if len(data) > max_bytes:
        raise AudioValidationError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    # Verify WAV containers immediately; compressed formats are checked by the decoder.
    if suffix == ".wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                if wav.getnframes() <= 0 or wav.getframerate() <= 0 or wav.getnchannels() <= 0:
                    raise AudioValidationError("WAV file has invalid audio dimensions.")
        except (wave.Error, EOFError) as exc:
            raise AudioValidationError("The WAV file is malformed or truncated.") from exc


def normalize_text(text: str, *, casefold_english: bool = True, remove_punctuation: bool = True) -> str:
    """Normalize Unicode and whitespace while preserving Devanagari and token boundaries."""
    text = unicodedata.normalize("NFC", text)
    if remove_punctuation:
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    if casefold_english:
        text = text.casefold()
    return " ".join(text.split())


def _edit_counts(ref: list[Any], hyp: list[Any]) -> tuple[int, int, int, list[tuple[str, Any, Any]]]:
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows): cost[i][0], back[i][0] = i, "D"
    for j in range(1, cols): cost[0][j], back[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            options = [(cost[i-1][j-1] + (ref[i-1] != hyp[j-1]), "M" if ref[i-1] == hyp[j-1] else "S"),
                       (cost[i-1][j] + 1, "D"), (cost[i][j-1] + 1, "I")]
            cost[i][j], back[i][j] = min(options, key=lambda x: x[0])
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    aligned: list[tuple[str, Any, Any]] = []
    while i or j:
        op = back[i][j]
        if op in ("M", "S"):
            if op == "S": subs += 1
            aligned.append((op, ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif op == "D":
            dels += 1; aligned.append((op, ref[i-1], None)); i -= 1
        else:
            ins += 1; aligned.append((op, None, hyp[j-1])); j -= 1
    aligned.reverse()
    return subs, dels, ins, aligned


def score_transcripts(reference: str, hypothesis: str, *, casefold_english: bool = True,
                      remove_punctuation: bool = True) -> dict[str, Any]:
    ref = normalize_text(reference, casefold_english=casefold_english, remove_punctuation=remove_punctuation)
    hyp = normalize_text(hypothesis, casefold_english=casefold_english, remove_punctuation=remove_punctuation)
    if not ref:
        raise ValueError("Reference transcript must contain at least one character.")
    rt, ht = ref.split(), hyp.split()
    subs, dels, ins, aligned = _edit_counts(rt, ht)
    char_sub, char_del, char_ins, _ = _edit_counts(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return {
        "reference_normalized": ref, "hypothesis_normalized": hyp,
        "reference_words": len(rt), "substitutions": subs, "deletions": dels, "insertions": ins,
        "wer": (subs + dels + ins) / len(rt),
        "cer": (char_sub + char_del + char_ins) / max(1, len(ref.replace(" ", ""))),
        "alignment": [{"operation": op, "reference": r, "hypothesis": h} for op, r, h in aligned],
        "note": "WER/CER use normalized text; timestamps are not inferred from transcript-only alignment.",
    }


def decode_audio(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode uploaded media to mono float32 using soundfile, then librosa fallback."""
    validate_audio(data, filename)
    if Path(filename).suffix.lower() == ".wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                channels, width, sr = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
                frames = wav.readframes(wav.getnframes())
            if width not in (1, 2, 3, 4):
                raise AudioValidationError("WAV sample width is not supported.")
            if width == 1:
                integers = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128
                scale = 128.0
            elif width == 3:
                raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
                values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
                integers = ((values ^ 0x800000) - 0x800000).astype(np.float32)
                scale = 8388608.0
            else:
                dtype = "<i2" if width == 2 else "<i4"
                integers = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                scale = float(2 ** (8 * width - 1))
            samples = integers.reshape(-1, channels).mean(axis=1) / scale
            return samples.astype(np.float32), int(sr)
        except (wave.Error, ValueError, EOFError) as exc:
            raise AudioValidationError("The WAV file could not be decoded.") from exc
    try:
        import soundfile as sf
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as first_exc:
        try:
            import librosa
            samples, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
            if samples.ndim == 1: samples = samples[None, :]
            samples = samples.T
        except Exception as exc:
            raise AudioValidationError("Could not decode this audio. MP3/M4A may require FFmpeg; install FFmpeg and reopen the app.") from exc
    return samples.mean(axis=1).astype(np.float32), int(sr)


def preprocess_audio(samples: np.ndarray, sample_rate: int, *, target_rate: int = 16000,
                     normalize: bool = False, trim_silence: bool = False) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Convert mono audio to target sample rate; optional peak normalization/trim."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(x) or sample_rate <= 0: raise ValueError("Audio and sample rate must be valid.")
    original_duration = len(x) / sample_rate
    if sample_rate != target_rate:
        import librosa
        x = librosa.resample(x, orig_sr=sample_rate, target_sr=target_rate).astype(np.float32)
    peak_before = float(np.max(np.abs(x)))
    if normalize and peak_before > 1e-8: x = x / peak_before * 0.95
    trim_samples = 0
    if trim_silence:
        import librosa
        x, indices = librosa.effects.trim(x, top_db=35)
        trim_samples = int(indices[0])
    info = {"original_duration_seconds": original_duration, "duration_seconds": len(x) / target_rate,
            "sample_rate": target_rate, "peak_before": peak_before, "trimmed_leading_samples": trim_samples}
    return x, target_rate, info


def waveform_points(samples: np.ndarray, max_points: int = 1800) -> list[float]:
    """Downsample waveform by peak buckets for a compact, readable chart."""
    x = np.asarray(samples).reshape(-1)
    if len(x) <= max_points: return x.astype(float).tolist()
    chunk = int(np.ceil(len(x) / max_points))
    return [float(np.max(np.abs(x[i:i+chunk])) * (1 if np.mean(x[i:i+chunk]) >= 0 else -1)) for i in range(0, len(x), chunk)]


def export_result(result: dict[str, Any], format: str) -> bytes:
    if format == "JSON": return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    if format == "TXT":
        return (f"Reference: {result.get('reference', '')}\nHypothesis: {result.get('hypothesis', '')}\n"
                f"WER: {result.get('wer', '')}\nCER: {result.get('cer', '')}\n").encode("utf-8")
    if format == "CSV":
        out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=["reference", "hypothesis", "wer", "cer", "substitutions", "deletions", "insertions"])
        writer.writeheader(); writer.writerow({k: result.get(k, "") for k in writer.fieldnames})
        return out.getvalue().encode("utf-8-sig")
    raise ValueError("Format must be TXT, CSV, or JSON.")


def benchmark_configurations(audio: np.ndarray, sample_rate: int, configs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run explicitly selected local faster-whisper model configurations and measure latency."""
    from faster_whisper import WhisperModel
    rows = []
    for config in configs:
        model_name = config["model"]
        import time
        start = time.perf_counter()
        model = WhisperModel(model_name, device="cpu", compute_type=config["compute_type"], cpu_threads=2)
        segments, meta = model.transcribe(audio, beam_size=1, vad_filter=True)
        segments = list(segments)
        rows.append({"model": model_name, "compute_type": config["compute_type"],
                     "latency_seconds": time.perf_counter()-start,
                     "language": meta.language, "text": "".join(s.text for s in segments).strip(),
                     "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]})
        del model
    return rows


@dataclass
class BenchmarkRecord:
    model: str
    compute_type: str
    latency_seconds: float
    language: str
    text: str
