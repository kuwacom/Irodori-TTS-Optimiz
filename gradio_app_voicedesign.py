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
    preferred = [
        path
        for path in candidates
        if "caption" in str(path).lower() or "voice_design" in str(path).lower()
    ]
    if preferred:
        return str(preferred[-1])
    if candidates:
        return str(candidates[-1])
    return "Aratako/Irodori-TTS-600M-v3-VoiceDesign"


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


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")
    if is_speaker_inversion_safetensors_path(checkpoint):
        raise ValueError("Speaker embedding files cannot be used as model checkpoints.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio-caption] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
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


def _describe_runtime(
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
    status = (
        "loaded model into memory" if reloaded else "model already loaded; reused existing runtime"
    )
    notes: list[str] = []
    if not runtime.model_cfg.use_caption_condition:
        notes.append(
            "warning: this checkpoint does not enable caption conditioning. Use gradio_app.py for reference-audio inference."
        )
    if runtime.model_cfg.use_speaker_condition_resolved:
        notes.append(
            "info: this checkpoint supports speaker conditioning; provide reference audio or keep no-reference enabled."
        )
    return "\n".join(
        [
            status,
            f"checkpoint: {runtime_key.checkpoint}",
            f"model_device: {runtime_key.model_device}",
            f"model_precision: {runtime_key.model_precision}",
            f"codec_device: {runtime_key.codec_device}",
            f"codec_precision: {runtime_key.codec_precision}",
            f"max_parallelism: {runtime.max_parallelism}",
            f"use_caption_condition: {runtime.model_cfg.use_caption_condition}",
            f"use_speaker_condition: {runtime.model_cfg.use_speaker_condition_resolved}",
            *notes,
        ]
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    text: str,
    caption: str,
    ref_wav: str | None,
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
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    speaker_kv_scale_raw: str,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
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

    text_value = "" if text is None else str(text).strip()
    caption_value = "" if caption is None else str(caption).strip()

    if text_value == "":
        raise ValueError("text is required.")

    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    seed = _parse_optional_int(seed_raw, "seed")
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    runtime, reloaded = get_cached_runtime(runtime_key, max_parallelism=int(max_parallelism))
    if not runtime.model_cfg.use_caption_condition:
        raise ValueError(
            "Loaded checkpoint does not enable caption conditioning. Use gradio_app.py for the original reference-audio model."
        )
    ref_wav_path = _resolve_ref_wav(ref_wav)
    effective_no_ref = ref_wav_path is None or not runtime.model_cfg.use_speaker_condition_resolved
    if effective_no_ref:
        ref_wav_path = None

    stdout_log(f"[gradio-caption] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio-caption] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "mode={} schedule={} sway_coeff={} seconds={} duration_scale={} steps={} seed={} candidates={}"
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
            requested_candidates,
        )
    )
    stdout_log(
        "[gradio-caption] conditioning: text={} caption={} speaker={}".format(
            "on" if text_value else "off",
            "on" if caption_value else "off (text-only)",
            "off (no-ref)" if effective_no_ref else "on",
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            caption=caption_value or None,
            ref_wav=ref_wav_path,
            ref_latent=None,
            no_ref=effective_no_ref,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=30.0,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            sampling_preset=str(sampling_preset),
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=0.0 if effective_no_ref else float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None if effective_no_ref else speaker_kv_scale,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
            lora_adapter=lora_adapter,
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs_voicedesign")
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
    stdout_log(f"[gradio-caption] saved {len(out_paths)} candidates")

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
    caption: str,
    ref_wav: str | None,
    num_steps: int,
    sampling_preset: str,
    seed_raw: str,
    t_schedule_mode: str,
    sway_coeff: float,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    speaker_kv_scale_raw: str,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    lora_adapter_raw: str,
    max_parallelism: int = 1,
    enable_watermark: bool = False,
    max_segment_seconds: float = 30.0,
    max_segment_chars: int = 180,
    chars_per_second: float = 10.0,
    min_segment_chars: int = 4,
    segment_gap_seconds: float = 0.2,
    segment_trim_silence_db: float = -40.0,
    max_batch_segments: int = 8,
    duration_scale: float = 1.0,
) -> tuple[object, ...]:
    """VoiceDesign版: 長文分割読み上げエントリポイント"""
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

    text_value = "" if text is None else str(text).strip()
    caption_value = "" if caption is None else str(caption).strip()

    if text_value == "":
        raise ValueError("text is required.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_text_len = _parse_optional_int(max_text_len_raw, "max_text_len")
    max_caption_len = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    seed = _parse_optional_int(seed_raw, "seed")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    runtime, reloaded = get_cached_runtime(runtime_key, max_parallelism=int(max_parallelism))
    if not runtime.model_cfg.use_caption_condition:
        raise ValueError(
            "Loaded checkpoint does not enable caption conditioning. Use gradio_app.py for the original reference-audio model."
        )
    ref_wav_path = _resolve_ref_wav(ref_wav)
    effective_no_ref = ref_wav_path is None or not runtime.model_cfg.use_speaker_condition_resolved
    if effective_no_ref:
        ref_wav_path = None

    stdout_log(f"[gradio-caption-long] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        f"[gradio-caption-long] request: long_text len={len(text_value)} "
        f"max_seg_seconds={max_segment_seconds} max_seg_chars={max_segment_chars} "
        f"gap={segment_gap_seconds}s trim_db={segment_trim_silence_db}"
    )
    stdout_log(
        "[gradio-caption-long] conditioning: text={} caption={} speaker={}".format(
            "on" if text_value else "off",
            "on" if caption_value else "off (text-only)",
            "off (no-ref)" if effective_no_ref else "on",
        )
    )

    # 事前分割プレビュー
    split_preview = split_long_text(
        text_value,
        max_seconds=max_segment_seconds,
        max_chars=max_segment_chars,
        chars_per_second=chars_per_second,
        min_segment_chars=min_segment_chars,
        duration_scale=float(duration_scale),
    )
    preview_lines = [f"segments: {len(split_preview.segments)} (wasSplit={split_preview.wasSplit})"]
    for idx, seg in enumerate(split_preview.segments):
        trunc = "..." if len(seg.text) > 60 else ""
        preview_lines.append(
            f"  [{idx}] '{seg.text[:60]}{trunc}' est={seg.estimatedSeconds:.1f}s"
        )

    result = runtime.synthesize_long(
        LongTextSamplingRequest(
            text=text_value,
            caption=caption_value or None,
            ref_wav=ref_wav_path,
            ref_latent=None,
            no_ref=effective_no_ref,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            max_ref_seconds=30.0,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            sampling_preset=str(sampling_preset),
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=0.0 if effective_no_ref else float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None if effective_no_ref else speaker_kv_scale,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
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

    out_dir = Path("gradio_outputs_voicedesign")
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
    stdout_log(f"[gradio-caption-long] saved combined + {len(result.segment_audios)} segment audios")

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

    with gr.Blocks(title="Irodori-TTS VoiceDesign Gradio") as demo:
        gr.Markdown("# Irodori-TTS VoiceDesign Inference")
        gr.Markdown(
            "VoiceDesign版モデル向けのUIです。caption を入れると caption / style conditioning、"
            "空欄なら text-only conditioning で推論します。"
        )

        # 共通設定: モデル・デバイス
        with gr.Row():
            checkpoint = gr.Textbox(
                label="Model Checkpoint (.pt/.safetensors or HF repo id)",
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
                        elem_id="irodori-voicedesign-text-input",
                    )
                    build_emoji_palette(text, open=False)
                caption = gr.Textbox(
                    label="Caption / Style Prompt (optional)",
                    lines=4,
                )
                ref_wav = gr.Audio(
                    label="Reference Audio Upload (optional, blank = no-reference mode)",
                    type="filepath",
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
                        cfg_scale_caption = gr.Slider(
                            label="CFG Scale Caption",
                            minimum=0.0,
                            maximum=10.0,
                            value=4.0,
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
                    cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional, set e.g. 3.0 for joint mode)", value="")
                    with gr.Row():
                        cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                        cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                        context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
                        speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                    with gr.Row():
                        max_text_len_raw = gr.Textbox(label="Max Text Len (optional)", value="")
                        max_caption_len_raw = gr.Textbox(label="Max Caption Len (optional)", value="")
                    with gr.Row():
                        truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                        rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                        rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
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
                        caption,
                        ref_wav,
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
                        cfg_scale_caption,
                        cfg_scale_speaker,
                        cfg_scale_raw,
                        cfg_min_t,
                        cfg_max_t,
                        context_kv_cache,
                        speaker_kv_scale_raw,
                        max_text_len_raw,
                        max_caption_len_raw,
                        truncation_factor_raw,
                        rescale_k_raw,
                        rescale_sigma_raw,
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
                        elem_id="irodori-voicedesign-long-text-input",
                    )
                    build_emoji_palette(long_text, open=False)
                long_caption = gr.Textbox(
                    label="Caption / Style Prompt (optional)",
                    lines=4,
                )
                long_ref_wav = gr.Audio(
                    label="Reference Audio Upload (optional, blank = no-reference mode)",
                    type="filepath",
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
                        long_cfg_scale_caption = gr.Slider(
                            label="CFG Scale Caption",
                            minimum=0.0,
                            maximum=10.0,
                            value=4.0,
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
                            value=30.0,
                            step=0.5,
                        )
                        long_max_segment_chars = gr.Slider(
                            label="Max Segment Chars",
                            minimum=50,
                            maximum=500,
                            value=180,
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
                            value=0.2,
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
                        long_speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                    with gr.Row():
                        long_max_text_len_raw = gr.Textbox(label="Max Text Len (optional)", value="")
                        long_max_caption_len_raw = gr.Textbox(label="Max Caption Len (optional)", value="")
                    with gr.Row():
                        long_truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                        long_rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                        long_rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
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
                        long_caption,
                        long_ref_wav,
                        long_num_steps,
                        long_sampling_preset,
                        long_seed_raw,
                        long_t_schedule_mode,
                        long_sway_coeff,
                        long_cfg_guidance_mode,
                        long_cfg_scale_text,
                        long_cfg_scale_caption,
                        long_cfg_scale_speaker,
                        long_cfg_scale_raw,
                        long_cfg_min_t,
                        long_cfg_max_t,
                        long_context_kv_cache,
                        long_speaker_kv_scale_raw,
                        long_max_text_len_raw,
                        long_max_caption_len_raw,
                        long_truncation_factor_raw,
                        long_rescale_k_raw,
                        long_rescale_sigma_raw,
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
            _describe_runtime,
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
    parser = argparse.ArgumentParser(
        description="Gradio app for caption-conditioned Irodori-TTS checkpoints."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    # 並列推論を有効にするため generate イベントの同時実行数を max_parallelism に合わせる
    # ユーザーが UI から並列数を変更した場合は、この値より多くのスレッドが
    # 同時に synthesize に入ることはない
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
