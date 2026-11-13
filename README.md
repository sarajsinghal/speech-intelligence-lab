# Speech Intelligence Lab

A local-first app for inspecting audio and evaluating Hindi, English, and code-switched Hinglish transcripts. It includes transcript normalization, WER/CER, edit alignment, local audio preprocessing, optional CPU transcription with faster-whisper, timestamps from the ASR model, exports, and measured configuration latency.

**Live app:** https://speech-intelligence-lab.streamlit.app/

## Deploying on Streamlit Community Cloud

This repository contains source code required to run the app. A public GitHub repository lets anyone inspect and copy the source. If source confidentiality matters, keep the repository private and deploy a private app with access granted only to intended viewers. Do not give viewers write access to the GitHub repository. An app hosted in the cloud processes uploaded recordings on the host, not on the viewer's computer.

The app starts in **Analyze audio** mode with no uploaded file or transcript example. No speech recordings or demo audio files are included. The Offline demo remains available from the mode selector and generates its synthetic waveform in memory.

1. Push this project folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the main file.
3. Deploy. Do not add speech recordings, model caches, `.venv`, secrets, or private data to the repository. Cloud uploads are processed by the hosted server; tell users before they upload sensitive recordings.
4. Optional faster-whisper transcription is not installed by the default requirements. Add it only if you intend to offer model downloads and accept their runtime/resource costs.

## What works

- **Offline demo by default:** synthetic waveform plus illustrative Hindi/Hinglish transcript pair; no audio file, network request, model, API key, GPU, or account required.
- Upload WAV, MP3, M4A, FLAC, and OGG up to 100 MB. WAV is validated at upload. MP3/M4A decoding may need FFmpeg.
- Local inspection, mono conversion, selectable resampling, optional peak normalization and silence trim. Normalization and trimming are off by default.
- Editable reference and hypothesis, word error rate (WER), character error rate (CER), substitutions/deletions/insertions, token alignment, TXT/CSV/JSON export.
- Explicitly optional faster-whisper CPU inference and small configuration benchmark. The app reports measured latency and model text; quality metrics only appear after comparison against a user-provided reference.
- ASR segment timestamps are available separately. Word alignment is text-based and does not pretend to assign timestamps to edit operations.
- Speaker diarization is not implemented.

## Windows 11 setup (PowerShell)

Use Python 3.10–3.12. From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. If script activation is blocked for this terminal, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again. The default demo is entirely offline.

To run automated tests:

```powershell
python -m pytest
```

### Optional local ASR

Only run this when you want model downloads and have enough disk space:

```powershell
python -m pip install faster-whisper
```

Choose **tiny / int8** first. The Whisper tiny model download is approximately 75 MB; `base` is roughly 145 MB and `small` roughly 480 MB, plus runtime packages/cache. The first run needs internet to fetch the model. This laptop's Ryzen 3 7320U and 8 GB RAM can run tiny on CPU, but inference may take minutes and larger choices can be slow or memory constrained. No GPU support is required or assumed. `float32` consumes more memory than `int8`.

For MP3/M4A decoding, install FFmpeg and ensure `ffmpeg.exe` is on PATH. WAV is the most dependable format. Decoding is performed locally.

## Evaluation choices

Text is normalized to Unicode NFC. Punctuation, when ignored, is replaced with a space so adjacent words do not merge. Whitespace is collapsed. English case-folding is on by default. Devanagari is preserved; there is no transliteration, stemming, spelling correction, or language-specific token rewriting. WER is word-token edit distance divided by reference words; CER is character edit distance divided by reference non-space characters. Empty references are rejected. Insertions can make WER exceed 100%.

This normalization is intentionally simple and reproducible. It does not resolve Hindi spelling variants, nukta conventions, Urdu script, English loanword spellings, or tokenizer disagreements. Record a chosen policy when comparing datasets. WER/CER are not semantic-quality or fairness scores.

## Architecture

```text
app.py (Streamlit UI)
  ├── src/speech_lab/core.py (validation, decoding, preprocessing, normalization,
  │                            edit-distance scoring, exports, CPU benchmark)
  ├── soundfile + librosa (local audio decode/resample/trim)
  └── optional faster-whisper + CTranslate2 (explicit CPU model inference)
```

No database or remote inference service is used. Uploaded data lives in app memory for the session. Optional model files are cached by faster-whisper outside the project. No audio sample is redistributed; the offline waveform is synthetic.

## Responsible use and limits

Get consent and follow applicable recording rules. Audio stays in the local app process, though explicit model use downloads model files from the hosting service. Remove sensitive files when done. Recognition varies with accents, dialects, code-switching, names, and noise; review outputs with a fluent speaker. WER alone misses semantic severity, intelligibility, speaker identity, and other important qualities. The app does not perform diarization, emotion detection, or semantic quality assessment.

## Resume bullet suggestions (only after you run it)

- Built a local-first Streamlit speech evaluation lab with Hindi/English text normalization, WER/CER scoring, edit alignment, and exportable results.
- Implemented CPU-oriented audio preprocessing and optional faster-whisper inference with measured latency and segment timestamp review.
- Added automated unit coverage for metric calculations, text normalization, WAV upload validation, and export formats.

Describe ASR as optional/integrated only if you installed it and verified it on your own recordings. Do not claim production accuracy, diarization, or benchmark quality without a documented evaluation corpus and results.
