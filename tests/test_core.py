import io
import wave

import pytest

from src.speech_lab.core import (AudioValidationError, export_result, normalize_text,
                                 score_transcripts, validate_audio)


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b"\0\0" * 160)
    return out.getvalue()


def test_normalization_preserves_devanagari_and_hinglish_tokens():
    assert normalize_text("  नमस्ते,  HELLO!  ") == "नमस्ते hello"
    assert normalize_text("A—B", remove_punctuation=True) == "a b"


def test_wer_cer_and_alignment_counts():
    result = score_transcripts("the cat sat", "the bat")
    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert result["wer"] == pytest.approx(2 / 3)
    assert result["cer"] > 0
    assert [x["operation"] for x in result["alignment"]] == ["M", "S", "D"]


def test_empty_reference_rejected():
    with pytest.raises(ValueError, match="Reference"):
        score_transcripts("  ", "hello")


def test_validate_wav_and_reject_extension_and_empty():
    validate_audio(wav_bytes(), "sample.WAV")
    with pytest.raises(AudioValidationError, match="Unsupported"):
        validate_audio(b"anything", "sample.exe")
    with pytest.raises(AudioValidationError, match="empty"):
        validate_audio(b"", "sample.wav")


def test_validate_malformed_wav_and_limit():
    with pytest.raises(AudioValidationError, match="malformed"):
        validate_audio(b"not a wav", "sample.wav")
    with pytest.raises(AudioValidationError, match="exceeds"):
        validate_audio(wav_bytes(), "sample.wav", max_bytes=10)


def test_exports_are_valid():
    result = {"reference": "नमस्ते", "hypothesis": "hello", "wer": .5, "cer": .2}
    assert b"Reference:" in export_result(result, "TXT")
    assert b"reference,hypothesis" in export_result(result, "CSV")
    assert "नमस्ते" in export_result(result, "JSON").decode("utf-8")

