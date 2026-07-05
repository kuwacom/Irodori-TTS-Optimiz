#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette
from irodori_tts.inference_runtime import (
    LongTextSamplingRequest,
    LongTextSamplingResult,
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)
from irodori_tts.long_text_splitter import split_long_text
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path

MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *(
                path
                for path in Path(".").glob("**/checkpoint_*.safetensors")
                if not is_speaker_inversion_safetensors_path(path)
            ),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-500M-v3"
    return str(candidates[-1])


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    # Gradioがcheckbox等のbool値を文字列として渡すのを安全に除外
    if text.lower() in {"true", "false"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    # Gradioがcheckbox等のbool値を文字列として渡すのを安全に除外
    if text.lower() in {"true", "false"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _parse_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null", "off", "disable", "disabled", "base"}:
        return None
    return text


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _resolve_ref_wav(uploaded_audio: str | None) -> str | None:
    if uploaded_audio is not None and str(uploaded_audio).strip() != "":
        return str(uploaded_audio)
    return None


def _coerce_gradio_file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _resolve_speaker_embedding(
    uploaded_embedding: object,
    speaker_embedding_path_raw: str | None,
) -> str | None:
    uploaded_path = _coerce_gradio_file_path(uploaded_embedding)
    raw_path = None
    if speaker_embedding_path_raw is not None and str(speaker_embedding_path_raw).strip():
        raw_path = str(speaker_embedding_path_raw).strip()
    if uploaded_path is not None and raw_path is not None:
        raise ValueError("Use either speaker embedding upload or speaker embedding path, not both.")
    return uploaded_path if uploaded_path is not None else raw_path


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool = False,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
        enable_watermark=bool(enable_watermark),
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    max_parallelism: int = 1,
    enable_watermark: bool = False,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=bool(enable_watermark),
    )
    runtime, reloaded = get_cached_runtime(runtime_key, max_parallelism=int(max_parallelism))
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}\n"
        f"max_parallelism: {runtime.max_parallelism}"
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    uploaded_audio: str | None,
    uploaded_speaker_embedding: object,
    speaker_embedding_path_raw: str,
    num_steps: int,
    sampling_preset: str,
    num_candidates: int,
    seed_raw: str,
    seconds_raw: str,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    lora_adapter_raw: str,
    max_parallelism: int = 1,
    enable_watermark: bool = False,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=bool(enable_watermark),
    )

    if str(text).strip() == "":
        raise ValueError("text is required.")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)
    speaker_embedding = _resolve_speaker_embedding(
        uploaded_embedding=uploaded_speaker_embedding,
        speaker_embedding_path_raw=speaker_embedding_path_raw,
    )
    if ref_wav is not None and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref = ref_wav is None and speaker_embedding is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key, max_parallelism=int(max_parallelism))
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "mode={} schedule={} sway_coeff={} seconds={} duration_scale={} steps={} seed={} no_ref={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            cfg_guidance_mode,
            t_schedule_mode,
            sway_coeff,
            "auto" if manual_seconds is None else manual_seconds,
            duration_scale,
            num_steps,
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )
    if speaker_embedding is not None:
        stdout_log(f"[gradio] speaker_embedding: {speaker_embedding}")

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            ref_wav=ref_wav,
            ref_latent=None,
            ref_embed=speaker_embedding,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=30.0,
            max_text_len=None,
            sampling_preset=str(sampling_preset),
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
            lora_adapter=lora_adapter,
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_paths: list[str] = []
    for i, audio in enumerate(result.audios, start=1):
        out_path = save_wav(
            out_dir / f"sample_{stamp}_{i:03d}.wav",
            audio.float(),
            result.sample_rate,
        )
        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio] saved {len(out_paths)} candidates")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))
    return (*audio_updates, detail_text, timing_text)




def _run_long_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    uploaded_audio: str | None,
    uploaded_speaker_embedding: object,
    speaker_embedding_path_raw: str,
    num_steps: int,
    sampling_preset: str,
    seed_raw: str,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    lora_adapter_raw: str,
    max_parallelism: int = 1,
    enable_watermark: bool = False,
    max_segment_seconds: float = 28.0,
    max_segment_chars: int = 200,
    chars_per_second: float = 10.0,
    min_segment_chars: int = 4,
    segment_gap_seconds: float = 0.15,
    segment_trim_silence_db: float = -40.0,
    max_batch_segments: int = 8,
    duration_scale: float = 1.0,
) -> tuple[object, ...]:
    """長文分割読み上げエントリポイント"""
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=bool(enable_watermark),
    )

    if str(text).strip() == "":
        raise ValueError("text is required.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)
    speaker_embedding = _resolve_speaker_embedding(
        uploaded_embedding=uploaded_speaker_embedding,
        speaker_embedding_path_raw=speaker_embedding_path_raw,
    )
    if ref_wav is not None and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker embedding are mutually exclusive.")
    no_ref = ref_wav is None and speaker_embedding is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key, max_parallelism=int(max_parallelism))
    stdout_log(f"[gradio-long] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        f"[gradio-long] request: long_text len={len(text)} max_seg_seconds={max_segment_seconds} "
        f"max_seg_chars={max_segment_chars} gap={segment_gap_seconds}s trim_db={segment_trim_silence_db}"
    )
    if speaker_embedding is not None:
        stdout_log(f"[gradio-long] speaker_embedding: {speaker_embedding}")

    # 事前分割プレビュー
    split_preview = split_long_text(
        text,
        max_seconds=max_segment_seconds,
        max_chars=max_segment_chars,
        chars_per_second=chars_per_second,
        min_segment_chars=min_segment_chars,
    )
    preview_lines = [f"segments: {len(split_preview.segments)} (wasSplit={split_preview.wasSplit})"]
    for idx, seg in enumerate(split_preview.segments):
        trunc = "..." if len(seg.text) > 60 else ""
        preview_lines.append(f"  [{idx}] '{seg.text[:60]}{trunc}' est={seg.estimatedSeconds:.1f}s")

    result = runtime.synthesize_long(
        LongTextSamplingRequest(
            text=str(text),
            ref_wav=ref_wav,
            ref_latent=None,
            ref_embed=speaker_embedding,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=1,
            decode_mode="sequential",
            max_ref_seconds=30.0,
            max_text_len=None,
            sampling_preset=str(sampling_preset),
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
            lora_adapter=lora_adapter,
            max_segment_seconds=float(max_segment_seconds),
            max_segment_chars=int(max_segment_chars),
            chars_per_second=float(chars_per_second),
            min_segment_chars=int(min_segment_chars),
            segment_gap_seconds=float(segment_gap_seconds),
            segment_trim_silence_db=float(segment_trim_silence_db),
            max_batch_segments=int(max_batch_segments),
            duration_scale=float(duration_scale),
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 全体音声を保存
    out_path = save_wav(
        out_dir / f"long_{stamp}_001.wav",
        result.audio.float(),
        result.sample_rate,
    )

    # セグメントごとにも保存
    seg_paths: list[str] = [str(out_path)]
    for i, seg_audio in enumerate(result.segment_audios, start=1):
        seg_path = save_wav(
            out_dir / f"long_{stamp}_seg{i:03d}.wav",
            seg_audio.float(),
            result.sample_rate,
        )
        seg_paths.append(str(seg_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"segments: {len(result.segments)}",
        f"total_combined_duration: {result.audio.shape[-1] / result.sample_rate:.2f}s",
        *[f"saved: {path}" for path in seg_paths],
        "",
        "=== Split Preview ===",
        *preview_lines,
        "",
        *result.messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio-long] saved combined + {len(result.segment_audios)} segment audios")

    # 出力: 全体音声1つ + 詳細 + タイミング
    return (str(out_path), detail_text, timing_text)

def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Gradio") as demo:
        gr.Markdown("# Irodori-TTS Inference (Cached Runtime)")
        gr.Markdown(
            "When settings are unchanged, runtime is reused and only sampling/decoding runs."
        )

        # 共通設定: モデル・デバイス
        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )
            max_parallelism = gr.Slider(
                label="Max Parallelism",
                minimum=1,
                maximum=8,
                value=1,
                step=1,
                scale=1,
            )
            enable_watermark = gr.Checkbox(label="Enable Watermark", value=False)

        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        ## 推論タブ
        with gr.Tabs():
            
            # タブ1: 通常推論
            with gr.Tab("Normal Synthesis"):
                with gr.Column():
                    text = gr.Textbox(
                        label="Text",
                        lines=6,
                        elem_id="irodori-text-input",
                    )
                    build_emoji_palette(text, open=False)
                with gr.Tabs():
                    with gr.Tab("Reference Audio"):
                        uploaded_audio = gr.Audio(
                            label="Reference Audio Upload (optional)",
                            type="filepath",
                        )
                    with gr.Tab("Speaker Embedding"):
                        with gr.Row():
                            uploaded_speaker_embedding = gr.File(
                                label="Speaker Embedding Upload (.speaker.safetensors, optional)",
                                type="filepath",
                                file_count="single",
                                scale=1,
                            )
                            speaker_embedding_path_raw = gr.Textbox(
                                label="Speaker Embedding Path (.speaker.safetensors, optional)",
                                value="",
                                scale=1,
                            )

                with gr.Accordion("Sampling", open=True):
                    with gr.Row():
                        num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                        sampling_preset = gr.Dropdown(
                            label="Sampling Preset",
                            choices=["custom", "quality", "balanced", "speed", "extreme"],
                            value="custom",
                        )
                        num_candidates = gr.Slider(
                            label="Num Candidates",
                            minimum=1,
                            maximum=MAX_GRADIO_CANDIDATES,
                            value=1,
                            step=1,
                        )
                        seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
                        seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value="")
                        duration_scale = gr.Slider(
                            label="Duration Scale",
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.01,
                        )

                    with gr.Row():
                        t_schedule_mode = gr.Dropdown(
                            label="Time Schedule",
                            choices=["linear", "sway"],
                            value="linear",
                        )
                        sway_coeff = gr.Slider(
                            label="Sway Coeff",
                            minimum=-1.0,
                            maximum=1.5,
                            value=-1.0,
                            step=0.1,
                            interactive=False,
                        )

                    with gr.Row():
                        cfg_guidance_mode = gr.Dropdown(
                            label="CFG Guidance Mode",
                            choices=["independent", "joint", "alternating"],
                            value="independent",
                        )
                        cfg_scale_text = gr.Slider(
                            label="CFG Scale Text",
                            minimum=0.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.1,
                        )
                        cfg_scale_speaker = gr.Slider(
                            label="CFG Scale Speaker",
                            minimum=0.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1,
                        )

                with gr.Accordion("Advanced (Optional)", open=False):
                    cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
                    with gr.Row():
                        cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                        cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                        context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
                    with gr.Row():
                        truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                        rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                        rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
                    with gr.Row():
                        speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                        speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                        speaker_kv_max_layers_raw = gr.Textbox(
                            label="Speaker KV Max Layers (optional)", value=""
                        )
                    lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")

                generate_btn = gr.Button("Generate", variant="primary")

                out_audios: list[gr.Audio] = []
                num_rows = (
                    MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
                ) // GRADIO_AUDIO_COLS_PER_ROW
                with gr.Column():
                    for row_idx in range(num_rows):
                        with gr.Row():
                            for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                                i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                                if i >= MAX_GRADIO_CANDIDATES:
                                    break
                                out_audios.append(
                                    gr.Audio(
                                        label=f"Generated Audio {i + 1}",
                                        type="filepath",
                                        interactive=False,
                                        visible=(i == 0),
                                        min_width=160,
                                    )
                                )
                out_log = gr.Textbox(label="Run Log", lines=8)
                out_timing = gr.Textbox(label="Timing", lines=8)

                generate_btn.click(
                    _run_generation,
                    inputs=[
                        checkpoint,
                        model_device,
                        model_precision,
                        codec_device,
                        codec_precision,
                        text,
                        uploaded_audio,
                        uploaded_speaker_embedding,
                        speaker_embedding_path_raw,
                        num_steps,
                        sampling_preset,
                        num_candidates,
                        seed_raw,
                        seconds_raw,
                        duration_scale,
                        t_schedule_mode,
                        sway_coeff,
                        cfg_guidance_mode,
                        cfg_scale_text,
                        cfg_scale_speaker,
                        cfg_scale_raw,
                        cfg_min_t,
                        cfg_max_t,
                        context_kv_cache,
                        truncation_factor_raw,
                        rescale_k_raw,
                        rescale_sigma_raw,
                        speaker_kv_scale_raw,
                        speaker_kv_min_t_raw,
                        speaker_kv_max_layers_raw,
                        lora_adapter_raw,
                        max_parallelism,
                        enable_watermark,
                    ],
                    outputs=[*out_audios, out_log, out_timing],
                )
                t_schedule_mode.change(
                    _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
                )

            # タブ2: 長文分割読み上げ
            with gr.Tab("Long Text Synthesis"):
                gr.Markdown(
                    "長文テキストを自動的に区切り、各セグメントを diffusion 最大時間内に収めて推論し、"
                    "結果を結合して1つの音声にするモードです。"
                )

                with gr.Column():
                    long_text = gr.Textbox(
                        label="Text (Long)",
                        lines=10,
                        elem_id="irodori-long-text-input",
                    )
                    build_emoji_palette(long_text, open=False)

                with gr.Tabs():
                    with gr.Tab("Reference Audio"):
                        long_uploaded_audio = gr.Audio(
                            label="Reference Audio Upload (optional)",
                            type="filepath",
                        )
                    with gr.Tab("Speaker Embedding"):
                        with gr.Row():
                            long_uploaded_speaker_embedding = gr.File(
                                label="Speaker Embedding Upload (.speaker.safetensors, optional)",
                                type="filepath",
                                file_count="single",
                                scale=1,
                            )
                            long_speaker_embedding_path_raw = gr.Textbox(
                                label="Speaker Embedding Path (.speaker.safetensors, optional)",
                                value="",
                                scale=1,
                            )

                with gr.Accordion("Sampling", open=True):
                    with gr.Row():
                        long_num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                        long_sampling_preset = gr.Dropdown(
                            label="Sampling Preset",
                            choices=["custom", "quality", "balanced", "speed", "extreme"],
                            value="custom",
                        )
                        long_seed_raw = gr.Textbox(label="Seed (blank=random)", value="")
                        long_duration_scale = gr.Slider(
                            label="Duration Scale",
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.01,
                        )

                    with gr.Row():
                        long_t_schedule_mode = gr.Dropdown(
                            label="Time Schedule",
                            choices=["linear", "sway"],
                            value="linear",
                        )
                        long_sway_coeff = gr.Slider(
                            label="Sway Coeff",
                            minimum=-1.0,
                            maximum=1.5,
                            value=-1.0,
                            step=0.1,
                            interactive=False,
                        )

                    with gr.Row():
                        long_cfg_guidance_mode = gr.Dropdown(
                            label="CFG Guidance Mode",
                            choices=["independent", "joint", "alternating"],
                            value="independent",
                        )
                        long_cfg_scale_text = gr.Slider(
                            label="CFG Scale Text",
                            minimum=0.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.1,
                        )
                        long_cfg_scale_speaker = gr.Slider(
                            label="CFG Scale Speaker",
                            minimum=0.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1,
                        )

                with gr.Accordion("Long Text Settings", open=True):
                    with gr.Row():
                        long_max_segment_seconds = gr.Slider(
                            label="Max Segment Seconds",
                            minimum=5.0,
                            maximum=30.0,
                            value=28.0,
                            step=0.5,
                        )
                        long_max_segment_chars = gr.Slider(
                            label="Max Segment Chars",
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=10,
                        )
                        long_chars_per_second = gr.Slider(
                            label="Chars per Second (estimate)",
                            minimum=5.0,
                            maximum=20.0,
                            value=10.0,
                            step=0.5,
                        )
                    with gr.Row():
                        long_min_segment_chars = gr.Slider(
                            label="Min Segment Chars",
                            minimum=1,
                            maximum=20,
                            value=4,
                            step=1,
                        )
                        long_segment_gap_seconds = gr.Slider(
                            label="Segment Gap (seconds)",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.15,
                            step=0.05,
                        )
                        long_segment_trim_silence_db = gr.Slider(
                            label="Segment Trim Silence (dB)",
                            minimum=-80.0,
                            maximum=-10.0,
                            value=-40.0,
                            step=1.0,
                        )

                        long_max_batch_segments = gr.Slider(
                            label="Max Batch Segments",
                            minimum=1,
                            maximum=32,
                            value=8,
                            step=1,
                        )
                with gr.Accordion("Advanced (Optional)", open=False):
                    long_cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
                    with gr.Row():
                        long_cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                        long_cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                        long_context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
                    with gr.Row():
                        long_truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                        long_rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                        long_rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
                    with gr.Row():
                        long_speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                        long_speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                        long_speaker_kv_max_layers_raw = gr.Textbox(
                            label="Speaker KV Max Layers (optional)", value=""
                        )
                    long_lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value="")

                long_generate_btn = gr.Button("Generate (Long Text)", variant="primary")

                long_out_audio = gr.Audio(
                    label="Combined Audio",
                    type="filepath",
                    interactive=False,
                )
                long_out_log = gr.Textbox(label="Run Log", lines=12)
                long_out_timing = gr.Textbox(label="Timing", lines=8)

                long_generate_btn.click(
                    _run_long_generation,
                    inputs=[
                        checkpoint,
                        model_device,
                        model_precision,
                        codec_device,
                        codec_precision,
                        long_text,
                        long_uploaded_audio,
                        long_uploaded_speaker_embedding,
                        long_speaker_embedding_path_raw,
                        long_num_steps,
                        long_sampling_preset,
                        long_seed_raw,
                        long_t_schedule_mode,
                        long_sway_coeff,
                        long_cfg_guidance_mode,
                        long_cfg_scale_text,
                        long_cfg_scale_speaker,
                        long_cfg_scale_raw,
                        long_cfg_min_t,
                        long_cfg_max_t,
                        long_context_kv_cache,
                        long_truncation_factor_raw,
                        long_rescale_k_raw,
                        long_rescale_sigma_raw,
                        long_speaker_kv_scale_raw,
                        long_speaker_kv_min_t_raw,
                        long_speaker_kv_max_layers_raw,
                        long_lora_adapter_raw,
                        max_parallelism,
                        enable_watermark,
                        long_max_segment_seconds,
                        long_max_segment_chars,
                        long_chars_per_second,
                        long_min_segment_chars,
                        long_segment_gap_seconds,
                        long_segment_trim_silence_db,
                        long_max_batch_segments,
                        long_duration_scale,
                    ],
                    outputs=[long_out_audio, long_out_log, long_out_timing],
                )
                long_t_schedule_mode.change(
                    _on_t_schedule_mode_change, inputs=[long_t_schedule_mode], outputs=[long_sway_coeff]
                )

        # 共通デバイス変更ハンドラ
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )

        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                max_parallelism,
                enable_watermark,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo

def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    # 並列推論を有効にするためイベントの同時実行数上限を設定
    demo.queue(max_size=20, default_concurrency_limit=8)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=EMOJI_PALETTE_CSS,
    )


if __name__ == "__main__":
    main()
