from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.speech_lab.core import (
    DEMO_HYPOTHESIS,
    DEMO_REFERENCE,
    AudioValidationError,
    benchmark_configurations,
    decode_audio,
    export_result,
    preprocess_audio,
    score_transcripts,
    validate_audio,
    waveform_points,
)

st.set_page_config(
    page_title="Speech Intelligence Lab",
    page_icon=":material/graphic_eq:",
    layout="wide",
)

st.title("Speech intelligence lab")
st.caption("Explore speech, edit transcripts, and evaluate Hindi, English, and Hinglish locally.")

with st.sidebar:
    st.header("Your workspace")
    mode = st.radio("Choose a mode", ["Offline demo", "Analyze audio"], index=1, key="mode")
    st.divider()
    st.markdown("**Privacy first**")
    st.caption("Audio is processed by the computer hosting this app. In a cloud deployment, it is sent to that host for analysis.")
    st.markdown("**Languages**")
    st.caption("हिन्दी · English · Hinglish")

audio_name = "Offline synthetic example"
samples: np.ndarray | None = None
sample_rate: int | None = None
duration = 3.2
decoded_audio: bytes | None = None

if mode == "Analyze audio":
    uploaded = st.file_uploader(
        "Choose an audio recording",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        max_upload_size=100,
        help="Maximum upload size: 100 MB. WAV is the most dependable format.",
    )
    if uploaded is None:
        st.info("Upload an audio recording to see its waveform and begin transcript review.", icon=":material/upload_file:")
    else:
        decoded_audio = uploaded.getvalue()
        audio_name = uploaded.name
        try:
            validate_audio(decoded_audio, audio_name)
            with st.spinner("Reading audio on this computer…"):
                raw_audio, original_rate = decode_audio(decoded_audio, audio_name)
            st.audio(decoded_audio)
            st.caption(f"{audio_name} · {original_rate:,} Hz source · converted to mono for analysis")

            with st.expander("Adjust audio preparation", expanded=False):
                prep_left, prep_mid, prep_right = st.columns(3)
                target_rate = prep_left.selectbox("Analysis sample rate", [16000, 22050, 44100], index=0)
                normalize_audio = prep_mid.toggle("Peak normalize", value=False, help="Off by default to preserve the recording's original dynamics.")
                trim_silence = prep_right.toggle("Trim quiet edges", value=False, help="Trims leading and trailing silence locally.")

            samples, sample_rate, audio_info = preprocess_audio(
                raw_audio,
                original_rate,
                target_rate=target_rate,
                normalize=normalize_audio,
                trim_silence=trim_silence,
            )
            duration = audio_info["duration_seconds"]
            st.success(f"Audio ready · {duration:.2f} seconds at {sample_rate:,} Hz", icon=":material/check_circle:")
        except (AudioValidationError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Audio processing failed: {exc}")

if mode == "Offline demo":
    sr = 16000
    timeline = np.linspace(0, duration, int(duration * sr), endpoint=False)
    envelope = np.zeros_like(timeline)
    for start, end, strength in [(0.2, 0.65, 0.35), (0.85, 1.5, 0.55), (1.7, 2.3, 0.3), (2.45, 3.0, 0.48)]:
        active = (timeline >= start) & (timeline <= end)
        envelope[active] = strength * (0.5 + 0.5 * np.sin(np.linspace(0, np.pi, active.sum())))
    samples = envelope * np.sin(2 * np.pi * 220 * timeline)
    sample_rate = sr

overview, review, recognize, about = st.tabs(
    ["Overview", "Transcript review", "Speech recognition", "About & privacy"]
)

with overview:
    st.subheader("Session overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Audio duration", f"{duration:.2f} s")
    metric_cols[1].metric("Sample rate", f"{sample_rate or 0:,} Hz")
    metric_cols[2].metric("Channels", "Mono")
    metric_cols[3].metric("Source", "Synthetic demo" if mode == "Offline demo" else audio_name)

    st.subheader("Waveform")
    if samples is not None:
        points = waveform_points(samples)
        chart_data = pd.DataFrame({"Amplitude": points})
        st.line_chart(
            chart_data,
            height=220,
            color="#7c3aed",
            x_label="Sample window",
            y_label="Amplitude",
            alt="Audio waveform over sample windows",
        )
        if mode == "Offline demo":
            st.info("This is a synthetic tone, not a speech recording. The displayed transcript is an example for trying the scoring tools.", icon=":material/science:")
    else:
        st.caption("Upload audio in Analyze audio mode to display its waveform.")

    st.subheader("Quick guide")
    guide_cols = st.columns(3)
    guide_cols[0].markdown("**1 · Add audio**\n\nUpload a recording or try the offline demo.")
    guide_cols[1].markdown("**2 · Review words**\n\nEdit the reference and hypothesis transcripts.")
    guide_cols[2].markdown("**3 · Explore scores**\n\nCompare WER/CER and inspect each word alignment.")

with review:
    st.subheader("Transcript review")
    st.caption("Type or paste both transcripts, then score the hypothesis against your reference.")
    ref_default = DEMO_REFERENCE if mode == "Offline demo" else ""
    hyp_default = DEMO_HYPOTHESIS if mode == "Offline demo" else ""
    ref_col, hyp_col = st.columns(2)
    reference = ref_col.text_area("Reference transcript", value=ref_default, height=150, key="reference")
    hypothesis = hyp_col.text_area("Hypothesis transcript", value=hyp_default, height=150, key="hypothesis")

    with st.expander("Choose text normalization", expanded=False):
        norm_left, norm_right = st.columns(2)
        casefold = norm_left.toggle("Ignore English capitalization", value=True)
        punctuation = norm_right.toggle("Ignore punctuation", value=True)
        st.caption("Unicode NFC is applied and whitespace is collapsed. Devanagari and English tokens are preserved; no transliteration or spelling correction is done.")

    score_clicked = st.button("Score these transcripts", type="primary", icon=":material/analytics:")
    if score_clicked:
        try:
            st.session_state["score"] = score_transcripts(
                reference,
                hypothesis,
                casefold_english=casefold,
                remove_punctuation=punctuation,
            )
            st.session_state["scored_reference"] = reference
            st.session_state["scored_hypothesis"] = hypothesis
        except ValueError as exc:
            st.error(str(exc))

    score = st.session_state.get("score")
    if score:
        st.divider()
        st.subheader("Your evaluation")
        score_cols = st.columns(5)
        score_cols[0].metric("Word error rate", f"{score['wer']:.1%}", border=True)
        score_cols[1].metric("Character error rate", f"{score['cer']:.1%}", border=True)
        score_cols[2].metric("Substitutions", score["substitutions"], border=True)
        score_cols[3].metric("Deletions", score["deletions"], border=True)
        score_cols[4].metric("Insertions", score["insertions"], border=True)
        st.caption("WER can exceed 100% when there are many inserted words. These text-edit scores do not measure meaning or overall speech quality.")

        st.markdown("**Word-by-word alignment**")
        operation_filter = st.multiselect(
            "Show alignment types",
            ["Match", "Substitution", "Deletion", "Insertion"],
            default=["Match", "Substitution", "Deletion", "Insertion"],
        )
        op_names = {"M": "Match", "S": "Substitution", "D": "Deletion", "I": "Insertion"}
        alignment = pd.DataFrame(score["alignment"])
        if not alignment.empty:
            alignment["type"] = alignment["operation"].map(op_names)
            alignment = alignment[alignment["type"].isin(operation_filter)]
            st.dataframe(
                alignment[["type", "reference", "hypothesis"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "type": st.column_config.TextColumn("Edit type"),
                    "reference": st.column_config.TextColumn("Reference word"),
                    "hypothesis": st.column_config.TextColumn("Hypothesis word"),
                },
                alt="Filtered word-by-word transcript alignment",
            )
        else:
            st.info("No alignment rows match the selected types.")

        if mode == "Analyze audio" and samples is not None:
            st.caption("The word alignment compares text only. It does not assign timestamps to individual edits.")

        export_result_data = {
            "project": "Project 15 — Speech Intelligence Lab",
            "audio_file": audio_name,
            "reference": st.session_state.get("scored_reference", reference),
            "hypothesis": st.session_state.get("scored_hypothesis", hypothesis),
            **score,
        }
        st.markdown("**Download this evaluation**")
        export_cols = st.columns(3)
        for column, fmt in zip(export_cols, ("TXT", "CSV", "JSON")):
            column.download_button(
                f"Download {fmt}",
                data=export_result(export_result_data, fmt),
                file_name=f"speech_evaluation.{fmt.lower()}",
                mime={"TXT": "text/plain", "CSV": "text/csv", "JSON": "application/json"}[fmt],
                width="stretch",
            )

with recognize:
    st.subheader("Optional local speech recognition")
    st.warning(
        "This feature needs an optional package and a model download. The tiny model is about 75 MB; recognition can take several minutes on a CPU laptop with 8 GB RAM. Nothing downloads until you install the package and start transcription.",
        icon=":material/download:",
    )
    model_col, compute_col, language_col = st.columns(3)
    model = model_col.selectbox("Model size", ["tiny", "base", "small"], index=0, help="Start with tiny for this CPU-first laptop.")
    compute = compute_col.selectbox("CPU precision", ["int8", "int8_float32", "float32"], index=0)
    language = language_col.selectbox("Language hint", ["Auto detect", "Hindi", "English"], index=0)

    can_transcribe = mode == "Analyze audio" and samples is not None
    if not can_transcribe:
        st.info("Switch to Analyze audio and upload a recording before starting transcription.", icon=":material/info:")
    if st.button("Transcribe on this computer", icon=":material/record_voice_over:", disabled=not can_transcribe):
        try:
            from faster_whisper import WhisperModel

            started_at = time.perf_counter()
            with st.spinner("Loading model and transcribing on CPU…"):
                recognizer = WhisperModel(model, device="cpu", compute_type=compute, cpu_threads=2)
                lang_code = {"Hindi": "hi", "English": "en"}.get(language)
                raw_segments, info = recognizer.transcribe(samples, language=lang_code, beam_size=1, vad_filter=True)
                raw_segments = list(raw_segments)
            elapsed = time.perf_counter() - started_at
            text = " ".join(segment.text.strip() for segment in raw_segments).strip()
            st.session_state["hypothesis"] = text
            st.session_state["asr_segments"] = [
                {"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text.strip()}
                for segment in raw_segments
            ]
            st.success(f"Completed in {elapsed:.1f} seconds · detected language: {info.language} ({info.language_probability:.0%})")
            st.text_area("Recognized transcript (copy into hypothesis above)", value=text, height=120, key="recognized_transcript")
            st.markdown("**Recognized segments and timestamps**")
            st.dataframe(pd.DataFrame(st.session_state["asr_segments"]), hide_index=True, alt="Speech recognition segments with start and end timestamps")
        except ImportError:
            st.error("Install the optional package in PowerShell with: python -m pip install faster-whisper")
        except Exception as exc:
            st.error(f"Local transcription failed: {exc}")

    st.divider()
    st.subheader("Compare model configurations")
    st.caption("Latency is measured on your computer. No model accuracy is invented; compare each result with a reference transcript in the review tab.")
    benchmark_options = st.multiselect(
        "Configurations to run",
        ["tiny / int8", "tiny / float32", "base / int8"],
        default=[],
    )
    can_benchmark = mode == "Analyze audio" and samples is not None and bool(benchmark_options)
    if st.button("Run selected comparisons", disabled=not can_benchmark):
        try:
            configs = [
                {"model": option.split(" / ")[0], "compute_type": option.split(" / ")[1]}
                for option in benchmark_options
            ]
            with st.spinner("Running selected CPU configurations…"):
                st.session_state["benchmark"] = benchmark_configurations(samples, sample_rate, configs)
        except ImportError:
            st.error("Install the optional package in PowerShell with: python -m pip install faster-whisper")
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")

    for row in st.session_state.get("benchmark", []):
        with st.expander(f"{row['model']} / {row['compute_type']} · {row['latency_seconds']:.1f} seconds"):
            st.write(row["text"])
            st.dataframe(pd.DataFrame(row["segments"]), hide_index=True, alt="Benchmark transcript segments with timestamps")

with about:
    st.subheader("About this project")
    st.markdown("""
    This is a local-first speech evaluation workbench. It focuses on Hindi, English, and code-switched Hinglish, and includes transcript normalization, WER/CER, edit alignment, optional local recognition, and measured CPU latency.

    **Not implemented:** speaker diarization. Segment timestamps come from the recognition model; word-level edit alignment is text-only.
    """)
    st.subheader("Responsible use")
    st.markdown("""
    - Get informed consent before recording or analyzing a person's voice and follow applicable local rules.
    - Audio is processed by the computer hosting this app. When deployed to a cloud host, the recording is sent there for analysis. Explicit use of speech recognition downloads model files from the model host.
    - Avoid uploading sensitive recordings on a shared computer and delete recordings when you no longer need them.
    - Accents, dialects, code-switching, names, and noisy recordings can affect recognition. Have a fluent speaker review the result.
    - WER alone does not measure meaning, speaker identity, emotion, accessibility, or whether an error is consequential.
    """)
