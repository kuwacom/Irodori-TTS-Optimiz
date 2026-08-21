# Irodori-TTS-Optimiz

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)  
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-v4.1-Small-Demo)  
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**Irodori-TTS-Optimiz** は、[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) をベースにした改善フォークです。  
本家の v4 コードベースを取り込みながら、v2/v3 チェックポイント互換性を保ち、推論時の使い勝手と速度を中心に最適化しています。

## クイックサマリー

| 項目                 | 推奨                                                            |
| -------------------- | --------------------------------------------------------------- |
| モデル               | `Aratako/Irodori-TTS-v4.1-Small`                                |
| 新しい GPU (sm_70+)  | `uv sync --extra cu128` (torch 2.10 + torchao)                  |
| 旧 GPU (P40等 sm_61) | `uv sync --extra legacy-cuda` (torch 2.5.1 + cu118, 量子化不可) |
| 高速推論             | `--sampling-preset speed` (24 steps, joint CFG)                 |
| 最速推論             | `--sampling-preset extreme` (16 steps, joint CFG 2.0)           |
| 品質優先             | `--sampling-preset quality` (40 steps, independent CFG)         |
| 並列推論             | `max_parallelism=2` 以上で CUDA Stream を使った並列生成         |
| 長文読み上げ         | `--long-text` で自動分割 + バッチ推論 + 結合                    |
| ウォーターマーク     | `--enable-watermark` で付与 (default: 無効)                     |
| キャッシュ           | 参照音声の latent と condition encoder 出力を自動キャッシュ     |

> [!WARNING]
>
> - legacy-cuda 環境では torchao がインストールされないため、量子化モデルは利用できません
> - 並列推論と LoRA の組み合わせは完全安全ではありません。LoRA 使用時は並列数 1 を推奨します
> - `--long-text` は `--ref-wav` (単数) のみサポートします。`--ref-wavs` (複数) を指定した場合は先頭1件を使用します
> - キャッシュは `./cache/latent/` に保存されます。不要な場合は削除可能です

## このフォークの変更点

推論まわりの改善を中心に、次の変更を入れています。  
v4-Small（`Aratako/Irodori-TTS-v4.1-Small`）を中心に、v2/v3 チェックポイントでも動作します。

### Token / Caption 実長 Trim

推論時の text / caption token を、tokenizer の `max_text_len` / `max_caption_len` の上限ではなく、mask 上で実際に使われている末尾位置まで trim するようにしています。

- 対象: `InferenceRuntime.synthesize()` 経由の CLI / Gradio / ランタイム API
- text: `text_mask` の実長まで `text_ids` / `text_mask` を slice
- caption: v4-Small / VoiceDesign checkpoint で `caption_mask` の実長まで `caption_ids` / `caption_mask` を slice
- 空 caption: 互換性のため最小長 1 を維持しつつ mask を全 false にします
- 効果: 短文推論で text encoder（v4-Small の ModernBERT 共有 backbone を含む）/ caption encoder / diffusion joint attention の context 長を削減できます

README 上の default は checkpoint metadata または `256` ですが、実際の文章が短い場合は 256 token 分を常に処理しないため、短文・短い caption の連続生成で特に効きます。  
品質には基本的に影響しない想定です。

### Reference Audio Latent Cache

Reference Audio を使った推論時に、参照音声を DACVAE latent へエンコードした結果をキャッシュします。

- 保存先: `./cache/latent/`
- 対象: `ref_wav` を指定した reference audio 推論（単数参照音声のみ）
- 非対象: `--no-ref` 推論、`ref_latent` を直接指定する推論、VoiceDesign の no-reference 推論
- v4 の複数参照音声 (`ref_wavs`): 複数ファイルを結合するためディスクキャッシュ対象外です。単数 `ref_wav` のみキャッシュされます
- 効果: 同じ reference audio と同じ前処理設定で再推論する場合、reference audio の読み込み・DACVAE encode を省略できます。

キャッシュキーには主に以下が含まれます。

- reference audio ファイル内容の SHA-256
- codec repo / deterministic encode 設定
- codec sample rate / hop length
- model latent dim
- `max_ref_seconds`
- `ref_normalize_db`
- `ref_ensure_max`

そのため、同じ音声ファイルでも reference の最大秒数や正規化設定を変えると別キャッシュとして扱われます。  
キャッシュは生成物なので Git 管理対象外です。不要になった場合は `./cache/latent/` を削除してください。

### Condition Encoder Cache

Reference Audio latent cache に加えて、推論ランタイム内で speaker / caption の encoder 出力を LRU cache します。

- 保存場所: メモリ上の `InferenceRuntime` 内 LRU cache
- cache size: speaker / caption それぞれ最大 32 件
- speaker cache 対象: reference latent と mask から作る speaker encoded state
- caption cache 対象: caption text/token/mask から作る caption encoded state
- 効果: 同じ reference audio や同じ VoiceDesign caption を使った連続生成で、speaker encoder / caption encoder の再計算を省略できます

この cache は runtime unload 時に破棄されます。  
Reference Audio latent cache と違い disk には保存しません。

現時点では encoded state までの cache です。

### Sampling Preset / CFG 高速化

CLI / Gradio / `SamplingRequest` に sampling preset を追加しています。  
`custom` / `quality` は明示指定したパラメータを優先し、`balanced` / `speed` / `extreme` は速度寄りの設定へ展開されます。  
v4-Small でも v2/v3 checkpoint でも同様に動作します。

| Preset     | 主な設定                                                                    | 想定用途             |
| ---------- | --------------------------------------------------------------------------- | -------------------- |
| `custom`   | CLI / UI / API で指定した値をそのまま使用                                   | 手動調整             |
| `quality`  | 既存の品質寄り設定を維持                                                    | 品質優先             |
| `balanced` | `num_steps=30`, `cfg_guidance_mode=alternating`, `cfg_min_t=0.55`           | 品質と速度のバランス |
| `speed`    | `num_steps=24`, `cfg_guidance_mode=joint`, `cfg_scale=3.0`, `cfg_min_t=0.6` | 連続生成向け         |
| `extreme`  | `num_steps=16`, `cfg_guidance_mode=joint`, `cfg_scale=2.0`, `cfg_min_t=0.7` | 速度最優先           |

使用例:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --sampling-preset speed \
  --output-wav outputs/sample_speed.wav
```

`joint` CFG は複数条件をまとめた uncond pass にするため、`independent` より diffusion step あたりの forward 回数を減らせます。  
ただし CFG のかけ方が変わるため、`speed` / `extreme` は品質・話者性・スタイル再現とのトレードオフがあります。

### Context K/V Cache と Speaker K/V Scaling

diffusion sampling 中に固定される text / speaker / caption context の K/V projection を、sampling 前に per-layer cache として構築して使います。  
`--context-kv-cache` は default で有効です。  
v4-Small の ModernBERT 共有 encoder でも同様に動作します。

- 対象: diffusion joint attention の context K/V projection
- 効果: sampling step ごとの context K/V projection 再計算を省略できます
- 関連: `--speaker-kv-scale` で speaker K/V を一時的に強調できます

現時点では text / speaker / caption をまとめた context K/V cache です。  
同じ speaker や caption だけを複数リクエストで再利用するには、context K/V を条件ごとに分離する追加実装が必要です。

### Watermark 切り替え

upstream v4 では SilentCipher が利用可能な場合常に watermark を付与しますが、このフォークでは default で無効化し、必要時のみ有効にできます。  
これにより watermark 処理のオーバーヘッドを省き、連続生成のスループットを向上させます。(長いと一回当たり数百ms違う)

- 実装: `irodori_tts/watermark.py` の `SilentCipherWatermarker` クラスを使用
- default: `enable_watermark=False`（無効）
- 有効化: CLI で `--enable-watermark`、RuntimeKey で `enable_watermark=True`
- 効果: watermark 付与を省略することで生成->出力までのレイテンシを削減

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは" \
  --ref-wav path/to/reference.wav \
  --enable-watermark \
  --output-wav outputs/sample_watermarked.wav
```

### CUDA Environment

このフォークは v4 の依存関係に合わせ、`uv sync --extra cu128` で CUDA 12.8 系の PyTorch を使います。  
Tesla P40 など Pascal 世代 GPU / `sm_61` 系では `cu128` 版 PyTorch が対応しないため、`legacy-cuda` extra を使ってください。

```bash
uv sync --extra legacy-cuda
```

この extra は PyTorch / Torchaudio を `2.5.1` + `cu118` 系に固定します。  
詳細は下記 [Installation](#installation) セクションを参照してください。

### 並列推論 (Parallel Inference)

1つのモデル重みを VRAM に載せたまま、複数リクエストを同時に推論可能にする機能です。  
v4-Small でも v2/v3 checkpoint でも動作します。  
各リクエストは steps / CFG scale / speaker / caption などを完全に独立してカスタムできます。  
API サーバーなどで `asyncio.to_thread` 等により複数スレッドから `runtime.synthesize()` を同時に呼び出すことを想定しています。

#### アーキテクチャ

```
FastAPI / Gradio サーバー
  └─ InferenceRuntime (1つだけロード)
      ├─ ワーカープール (Semaphore で並列実行数制限)
      │   ├─ Worker 0: CUDA Stream 0 → model (共有重み)
      │   ├─ Worker 1: CUDA Stream 1 → model (同じ重み)
      │   └─ Worker N: CUDA Stream N → model (同じ重み)
      ├─ _lora_lock (LoRA アダプタ切り替え時のみ排他)
      └─ _InferenceScope (セマフォ/Stream/LoRA context の一括ライフタイム管理)
```

- モデル重みは1つだけ VRAM にロードし、全ワーカーで共有する
- 各ワーカーには専用の CUDA Stream が割り当てられ、GPU カーネルレベルで並列オーバーラップする
- LoRA アダプタの切り替えは排他制御下で行うが、推論本体は並列に実行される
- 動的バッチングは行わない（リクエストごとにパラメータが異なるため非現実的）

#### RoPE キャッシュの事前確保

並列推論では `_freqs_cis_cache` の上書き競合が問題になります。  
これを防ぐため、`InferenceRuntime` の初期化時に `prewarm_rope_cache()` を呼び出し、  
全エンコーダ (text / caption / speaker) の RoPE キャッシュを最大長で事前確保しています。

#### skip_timing_sync

並列数 2 以上の場合、タイミング計測の `torch.cuda.synchronize()` をスキップしてスループットを向上させます（`_skip_timing_sync` フラグ）。  
`total_to_decode` の計測精度は下がりますが、並列推論時のオーバーヘッドを削減できます。

#### 使い方

初期化時に `max_parallelism` を指定:

```python
from irodori_tts import InferenceRuntime, RuntimeKey

key = RuntimeKey(
    checkpoint="path/to/model.safetensors",
    model_device="cuda",
    model_precision="bf16",
)

# 初期化時に並列数を指定
runtime = InferenceRuntime.from_key(key, max_parallelism=3)
```

キャッシュ経由でも指定可能:

```python
from irodori_tts import get_cached_runtime

runtime, loaded = get_cached_runtime(key, max_parallelism=3)
```

実行時に動的に変更することも可能:

```python
# キャッシュヒット時や、サーバー稼働中に並列数を変更する場合
runtime.set_max_parallelism(4)
```

`set_max_parallelism()` はセマフォと CUDA Stream プールを再構築します。  
実行中のリクエストがある場合、それらの完了後に新しい並列数が反映されます。

#### Gradio Web UI

Gradio UI の Load Model 横に **Max Parallelism** スライダー (1〜8) を追加しています。  
モデルロード時に設定するほか、Generate ボタン押下時にも値が反映されます。

#### API サーバーからの利用

```python
import asyncio
from irodori_tts import InferenceRuntime, RuntimeKey, SamplingRequest

# 初期化 (1回だけ)
runtime = InferenceRuntime.from_key(key, max_parallelism=3)

# 複数リクエストを同時に処理
async def handle_request(text: str):
    req = SamplingRequest(text=text, no_ref=True, ...)
    # イベントループをブロックしないよう別スレッドで実行
    return await asyncio.to_thread(runtime.synthesize, req)
```

`runtime.synthesize()` はスレッドセーフです。  
内部でセマフォによる並列実行数制限と CUDA Stream 分離が行われるため、  
呼び出し側でロックを取得する必要はありません。

#### 注意点

- 並列実行数は VRAM に依存します。v4-Small モデルなので 24GB GPU で 3〜4 並列が目安です。
- 並列数を増やすと個別リクエストの所要時間は延びるが、スループット（単位時間あたりの処理リクエスト数）は向上します
- `_skip_timing_sync`: 並列数 2 以上の場合、タイミング計測の `torch.cuda.synchronize()` をスキップしてスループットを向上させる（`total_to_decode` の計測精度は下がる）

### 長文分割読み上げ (Long Text Synthesis)

diffusion の最大生成時間（30秒）を超える長文を、句読点・絵文字などの自然な区切りで分割し、  
バッチ推論して結合する機能です。`synthesize_long()` として通常推論とは独立に実装しています。  
v4-Small の `forward_with_encoded_conditions` と統合されており、共通条件（speaker / caption / CFG）を  
1回だけ前処理した上でバッチ推論を実行します。

#### 処理フロー

1. **テキスト分割** (`split_long_text`): 句読点・絵文字ベースで自然に分割。
   `duration_scale` を考慮し、実効上限秒数 (`max_segment_seconds / duration_scale`) に収まる文字数を逆算
2. **共通前処理**: speaker参照・caption・CFG等をバッチ/逐次ループの外で1回だけ計算（効率化とバグ防止）
3. **バッチ推論**: セグメントを `max_batch_segments` 件ずつチャンクに分け、各チャンクを1回の diffusion sampling で一括生成
4. **音声結合**: 各セグメントの前後無音をトリムし、`segment_gap_seconds` 分の無音を挿入して結合

#### 主なパラメータ

| パラメータ                  | デフォルト | 説明                                                                                              |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| `--max-segment-seconds`     | `30.0`     | 1セグメントの最大推定秒数                                                                         |
| `--max-segment-chars`       | `180`      | 1セグメントの最大文字数                                                                           |
| `--segment-gap`             | `0.2`      | セグメント間の無音秒数（前後無音トリム後）。ゆったりめのキャラクターでは `0.4` 程度に広げると自然 |
| `--segment-trim-silence-db` | `-40.0`    | 前後無音トリムのしきい値 (dB)                                                                     |
| `--max-batch-segments`      | `8`        | 1バッチで同時に処理するセグメント最大数                                                           |
| `--duration-scale`          | `1.0`      | 話速スケール（分割時の上限秒数にも反映）                                                          |

`max_batch_segments=1` にすると、従来の逐次処理と同等になります。

##### 推奨プリセット目安 (`duration_scale = 1.0` 前提)

`max_segment_seconds / max_segment_chars / segment_gap_seconds` の組み合わせは、  
読み上げの「詰め込み具合」と「自然さ」のトレードオフになります。  
以下は目安となる3つの代表例です:

| 設定                   | 秒数 / 文字 / gap | 特徴                                                                 |
| ---------------------- | ----------------- | -------------------------------------------------------------------- |
| **自然** (デフォルト)  | `30 / 180 / 0.2`  | 各セグメントを個別で作ってつなげるのに等しい、無理なく自然な読み上げ |
| **早め・詰め込み気味** | `30 / 200 / 0.2`  | 少々早めのテンポで、1セグメントに多く詰め込む読み上げスタイル        |
| **時間不足感あり**     | `28 / 200 / 0.2`  | セクション時間が足りていない感があり、長文では分割回数が増える       |

> **gap の調整**: `segment_gap_seconds` はキャラクターの話速テンポに合わせて調整してください。
> デフォルトの `0.2s` は標準的なテンポ向けです。
> ゆったりめのキャラクター(ゆっくり話す・間を多く取る話し方) の場合は `0.4s` 程度に広げると、セグメント間の休止が自然になり、全体として落ち着いた読み上げになります。

#### CLI

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "長いテキストを指定します。句読点で自然に分割され、バッチ推論されて結合されます。" \
  --ref-wav path/to/reference.wav \
  --long-text \
  --max-segment-seconds 30 \
  --max-batch-segments 4 \
  --segment-gap 0.2 \
  --output-wav outputs/long_sample.wav
```

#### Gradio Web UI

`gradio_app.py` / `gradio_app_voicedesign.py` に "Long Text Synthesis" タブを追加しています。  
通常推論タブと並存し、分割プレビュー・gap秒数・trimしきい値・batch数などをUIから調整できます。

#### API

```python
from irodori_tts import InferenceRuntime, RuntimeKey, LongTextSamplingRequest

runtime = InferenceRuntime.from_key(key)
result = runtime.synthesize_long(
    LongTextSamplingRequest(
        text="長いテキスト...",
        ref_wav="path/to/reference.wav",
        max_segment_seconds=30.0,
        max_batch_segments=8,
        segment_gap_seconds=0.2,
        segment_trim_silence_db=-40.0,
        duration_scale=1.0,
        num_steps=40,
    ),
)
# result.audio: 結合された全体音声
# result.segment_audios: セグメントごとの音声リスト
```

### 今後のtodo

- 生成結果 / reference latent 管理の改善
- condition K/V cache の text / speaker / caption 分離
- CFG cache の lazy build
- CFG 用 `x_t` buffer 再利用
- decode 前 latent trim
- tail trim 判定の vectorize

---

以下本家README拡張

Irodori-TTS は Flow Matching ベースの Text-to-Speech モデルです。  
アーキテクチャと学習設計は [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) に従い、[DACVAE](https://github.com/facebookresearch/dacvae) の連続 latent を生成対象としています。

> [!IMPORTANT]
> v4 コードベースをベースに、**Irodori-TTS-v4.1-Small** リリース向けに調整しています。
> v2/v3 チェックポイントの推論互換性も維持しています。
> v1 チェックポイントは v2/v3/v4 と互換性がありません。

OpenAI 互換推論 API サーバーは [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) を参照してください。  
モデル重みと音声サンプルは [Irodori-TTS-v4.1-Small model card](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) を参照してください。

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Multi-modal Voice Design**: v4-Small combines text, reference speech, and caption text for voice identity plus style/emotion control
- **Long Reference Audio**: One or more reference clips can be concatenated up to the checkpoint's 120-second limit
- **Emoji-based Style Control**: Emoji annotations in input text can influence delivery and non-verbal vocal expressions in supported checkpoints
- **Automatic Duration Prediction**: v4-Small estimates output length without manual `--seconds`
- **Automatic Watermarking**: Generated audio is watermarked with [SilentCipher](https://github.com/sony/silentcipher) when available
- **Multi-GPU Training**: Distributed training via `uv run --no-sync torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Speaker Inversion**: Learn reusable speaker embedding tokens for a target voice while freezing the base model
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The current release, **`Aratako/Irodori-TTS-v4.1-Small`**, unifies the previous base and  
VoiceDesign families in one checkpoint. It supports 3-branch conditioning from text,  
reference speech, and caption text. Released v2/v3 checkpoints remain supported for inference.

Shared building blocks:

1. **Shared Text/Caption Encoder**: A fine-tuned ModernBERT backbone processes both reading text and caption text
2. **Reference Latent Encoder**: Encodes patched reference audio latents for speaker identity conditioning, with up to 120 seconds of combined reference audio in v4-Small
3. **Condition Projectors**: Separate text and caption projectors map the shared encoder states into their conditioning spaces
4. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs
5. **Duration Predictor**: Integrated predictor for automatic output length estimation

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. The released v2/v3/v4 checkpoints use the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/kuwacom/Irodori-TTS-Optimiz.git
cd Irodori-TTS-Optimiz
uv sync --extra cu128  # NVIDIA CUDA 12.8 (Linux/Windows)
```

If you want to explicitly select a PyTorch backend, use one of the backend  
extras below:

```bash
# NVIDIA CUDA 12.8 on Linux/Windows
uv sync --extra cu128

# AMD ROCm on Linux/WSL
uv sync --extra rocm

# Intel XPU on Linux/Windows
uv sync --extra xpu

# CPU-only, or macOS CPU/MPS via PyPI
uv sync --extra cpu
```

The PyTorch backend extras are mutually exclusive. The `cu128` extra uses the  
PyTorch CUDA 12.8 index, the `rocm` extra uses the PyTorch ROCm index on  
Linux, and the `xpu` extra uses the PyTorch XPU index on Linux/Windows.  
The `cpu` extra uses the CPU PyTorch index on Linux/Windows and falls  
back to the standard PyPI PyTorch wheels on macOS.

After syncing with a backend extra, use `uv run --no-sync ...` for the commands  
below to avoid re-syncing the environment without the selected PyTorch backend  
extra.

The `rocm` extra includes `pytorch-triton-rocm` because `triton-rocm` alone does  
not provide `triton.language` for the `transformers` to `torch._dynamo` import  
path. This was validated with AMD GPU inference.

### Legacy NVIDIA GPU (Tesla P40 / sm_61)

PyTorch `cu128` builds do not support Pascal-generation GPUs such as `Tesla P40` (`sm_61`).  
Use the legacy CUDA extra on those machines:

```bash
uv sync --extra legacy-cuda
```

This extra pins PyTorch / Torchaudio to `2.5.1` from the `cu118` index, which is a better fit for older NVIDIA cards.

### Hugging Face cache path

If you want model downloads under this repository, set the cache directories before launching:

```bash
export HF_HOME="$PWD/models/hub"
export HUGGINGFACE_HUB_CACHE="$PWD/models/hub"
```

Then WebUI / CLI downloads such as `Aratako/Irodori-TTS-v4.1-Small` will be stored under `./models/hub`.

## Quick Start

### Simple Inference

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

Pure VoiceDesign from text + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

Style-controlled voice cloning with text + reference speech + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --ref-wav path/to/reference.wav \
  --caption "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。" \
  --output-wav outputs/sample_voice_design_clone.wav
```

Long-reference checkpoints can concatenate multiple reference clips in the specified order:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "複数の参照音声を使って合成します。" \
  --caption "落ち着いた自然な声" \
  --ref-wavs ref_01.wav ref_02.wav ref_03.wav \
  --output-wav outputs/sample_long_reference.wav
```

Each waveform is encoded independently before its latent is concatenated. The combined  
reference is trimmed to the checkpoint's maximum reference duration. Use `--ref-latents`  
in the same way for precomputed latent files.

For v4-Small, prefer multiple clean, shorter clips from the same speaker when using a long  
reference. The model was trained with randomly concatenated short utterances, and the measured  
speaker-similarity benefit used the same construction. A combined duration of approximately  
30 seconds already captured most of the measured gain. A single uninterrupted long recording  
is accepted by inference, but that input format has not been evaluated and may behave differently.

### Speaker Inversion Inference

Use a learned Speaker Inversion embedding instead of reference audio:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed path/to/my.speaker.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Gradio Web UI

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.  
The hosted v4-Small demo is available at [Aratako/Irodori-TTS-v4.1-Small-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-v4.1-Small-Demo).  
The reference input area accepts one or more audio files, which can be reordered before  
generation and are concatenated in the displayed order. For long-reference cloning, upload  
multiple clean, shorter clips from the same speaker; this matches v4-Small training. A single  
uninterrupted long recording is accepted but has not been evaluated. The standard UI also  
supports a Speaker Inversion embedding through the adjacent tab.

For VoiceDesign checkpoints, use the dedicated UI:

```bash
uv run --no-sync python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The same hosted v4-Small demo supports VoiceDesign and reference-audio conditioning.

For Tesla P40-class GPUs with a local Hugging Face cache:

```bash
HF_HOME="$PWD/models/hub" \
HUGGINGFACE_HUB_CACHE="$PWD/models/hub" \
uv run --no-sync --extra legacy-cuda python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

Both UIs default to `Aratako/Irodori-TTS-v4.1-Small`. `gradio_app_voicedesign.py` exposes  
caption conditioning, while `gradio_app.py` includes the Speaker Inversion input.  
Both UIs include a **Long Text Synthesis** tab alongside the normal synthesis tab, allowing you to try split-and-concatenate generation interactively.

## Inference

### CLI

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run --no-sync python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

v4-Small supports caption conditioning. It can run with  
caption only by passing `--no-ref`, or with both reference speech and caption by passing  
`--ref-wav`, `--ref-wavs`, `--ref-latent`, `--ref-latents`, or `--ref-embed`.

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。" \
  --caption "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_voice_design_ref_caption.wav
```

The older `Aratako/Irodori-TTS-500M-v2-VoiceDesign` checkpoint is still supported, but it is caption-only and intentionally ignores speaker/reference conditioning.

LoRA adapter directories can be loaded dynamically at inference time without  
exporting a merged checkpoint:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/base_model.safetensors \
  --lora-adapter outputs/irodori_tts_lora/checkpoint_final \
  --text "こんにちは、私はAIです。これはLoRA推論のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_lora.wav
```

Speaker Inversion embedding checkpoints can be used with the same base model that  
was used for inversion training. Pass the embedding with `--ref-embed`;  
it is mutually exclusive with `--ref-wav`, `--ref-latent`, and `--no-ref`.

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、私はAIです。これはSpeaker Inversion推論のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Output Duration

v4-Small integrates duration prediction into inference.  
When `--seconds` is omitted, the runtime estimates the output length from the input  
text and enabled conditions, then generates audio for that estimated duration. Use  
`--duration-scale` to multiply the predicted length (`>1` longer, `<1` shorter). For  
exact control, pass `--seconds` manually.

Older v2 checkpoints were trained with fixed-length 30-second targets. They remain  
supported by the current codebase and still accept manual `--seconds`, but forcing a  
non-default duration can reduce audio quality; prefer v4-Small for automatic  
or scaled duration control.

### Sway Sampling

For faster experimental inference, Sway Sampling can be combined with fewer Euler  
steps:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav outputs/sample_sway.wav
```

### Additional Inference Notes

For tuning guidance and detailed explanations of inference options, see the  
[Parameter Guide](docs/parameters.md).

Generated audio is passed through [SilentCipher](https://github.com/sony/silentcipher) watermarking automatically when the dependency and model files are available.

## Training

This section describes how to train **Irodori-TTS-v4.1-Small**. For training instructions  
for previous models, refer to the documentation in the corresponding version tags.

### 1. Prepare the Training Manifest

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

v4-Small learns from text, speaker/reference audio, and captions. Include `speaker_id` and  
`caption` where available so all three conditioning paths can be trained. A Speaker  
Inversion manifest does not require `speaker_id`, because the run learns one shared speaker  
embedding from the target-speaker samples.

The manifest `caption` value may also be a list of strings; training randomly selects one  
non-empty caption each time that row is loaded.

This produces a JSONL manifest with entries like:

```json
{
  "text": "こんにちは",
  "caption": "落ち着いた、近い距離感の女性話者",
  "latent_path": "data/latents/00001.pt",
  "speaker_id": "myorg/my_dataset:speaker_001",
  "num_frames": 750
}
```

### 2. Train v4-Small

Single-GPU training:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors
```

The v4-Small config trains the RF body, duration predictor, and shared pretrained text/caption  
backbone jointly. The duration predictor regresses `log1p(num_frames)` with Huber loss and  
uses the token-sum architecture selected from ablations. See the parameter guide for its  
architecture details.

Multi-GPU DDP training:

```bash
uv run --no-sync torchrun --nproc_per_node 4 train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run --no-sync python train.py --help` for all available options.  
For a more detailed explanation of model and training config fields, see [Parameter Guide](docs/parameters.md).

### 3. LoRA Fine-Tuning

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh. The duration predictor is kept as part of the saved adapter by default.

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors
```

The v4-Small LoRA config targets diffusion attention by default and saves the duration  
predictor with the adapter. To adapt the shared ModernBERT backbone, select the  
`pretrained_backbone_attn` or `pretrained_backbone_attn_mlp` target preset.

LoRA target presets, adapter saving behavior, and resume details are covered in the  
[Parameter Guide](docs/parameters.md).

### 4. Speaker Inversion

Speaker Inversion trains only a small set of speaker embedding tokens while keeping the  
base Irodori-TTS model frozen. It is useful when you want a reusable speaker identity  
checkpoint instead of providing reference audio at every inference call.

Prepare a manifest from the target speaker's audio, then initialize from v4-Small:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/name
```

The saved checkpoints are embedding-only `.speaker.safetensors` files, for example  
`outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors`. Use that file  
with the base model during inference:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

To continue from a saved embedding, set `speaker_inversion_init_embedding` in the  
config or pass `--speaker-inversion-init-embedding path/to/checkpoint.speaker.safetensors`.  
Full trainer `--resume` is intentionally not used for Speaker Inversion checkpoints.  
Enable `gradient_checkpointing: true` or pass `--gradient-checkpointing` if GPU memory is tight.

### 5. Resume Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 6. Convert a Training Checkpoint

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference. If you do not want to merge the adapter, pass the adapter directory directly to `infer.py --lora-adapter` or the matching Gradio field.

For checkpoints with a pretrained text encoder, conversion also writes a `tokenizer/`  
directory beside the safetensors file and embeds the encoder architecture config in the file.  
Keep the safetensors file and `tokenizer/` directory together when publishing or moving the model.

## Quantization

Quantized variants of Irodori-TTS reduce the memory required by the TTS model during  
inference. Pre-quantized v4-Small checkpoints are available from  
[Aratako/Irodori-TTS-v4.1-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized).  
Select a variant by appending its subdirectory name to the Hugging Face repository ID:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only \
  --model-precision bf16 \
  --text "こんにちは、私はAIです。" \
  --no-ref \
  --output-wav outputs/sample_int8.wav
```

Available schemes are `int8-weight-only` (W8A16), `int8-dynamic` (W8A8),  
`int4-weight-only` (W4A16, group size 128 by default), `float8-weight-only` (FP8 weights  
and BF16 activations), and `float8-dynamic` (FP8 weights and activations).  
INT4 weight-only uses the CUDA tinygemm kernel and requires compute capability 8.0 or newer.  
Only the selected model variant and its tokenizer assets are downloaded.

`--model-precision bf16` controls the unquantized layers and floating-point activations;  
quantized weights retain their stored quantization format.

To quantize another compatible inference checkpoint locally, use  
`quantize_checkpoint.py`. INT8 weight-only is the default:

```bash
uv run --no-sync python quantize_checkpoint.py path/to/model.safetensors \
  --quantization int8-weight-only \
  --output path/to/quantized/model.safetensors
```

The default `core` profile quantizes the attention and MLP weights in the text,  
speaker, and diffusion Transformer blocks. Projectors, AdaLN, duration prediction,  
and the codec remain unquantized. `--profile all-linear` is available for more aggressive  
experimentation.

Dynamic `--lora-adapter` inference is supported with quantized base checkpoints.  
Train the adapter against the matching full-precision base model.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # Gradio web UI for VoiceDesign checkpoints
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
├── quantize_checkpoint.py      # torchao checkpoint quantization
│
├── docs/
│   └── parameters.md         # Detailed parameter guide
│
├── irodori_tts/                # Core library
│   ├── model.py                # TextToLatentRFDiT architecture
│   ├── rf.py                   # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                # DACVAE codec wrapper
│   ├── dataset.py              # Dataset and collator
│   ├── duration.py             # Duration predictor utilities
│   ├── tokenizer.py            # Pretrained LLM tokenizer wrapper
│   ├── config.py               # Model and training config dataclasses
│   ├── inference_runtime.py    # Cached, thread-safe inference runtime
│   ├── long_text_splitter.py   # Long text segmentation for synthesis
│   ├── lora.py                 # PEFT LoRA integration helpers
│   ├── quantization.py         # torchao checkpoint serialization/load helpers
│   ├── watermark.py            # SilentCipher watermark wrapper
│   ├── speaker_inversion.py    # Speaker Inversion embedding save/load helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── gradio_emoji_palette.py # Emoji palette for Gradio UI
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_v4_small.yaml                    # Irodori-TTS-v4-Small training config
    ├── train_v4_small_lora.yaml               # v4-Small LoRA fine-tuning config
    ├── train_v4_small_speaker_inversion.yaml  # v4-Small Speaker Inversion config
    ├── train_500m_v3_phase1_body.yaml        # 500M v3 body training config
    ├── train_500m_v3_phase2_duration.yaml    # 500M v3 duration-predictor training config
    ├── train_500m_v3_voice_design_phase1_body.yaml     # 600M v3 VoiceDesign body config
    ├── train_500m_v3_voice_design_phase2_duration.yaml # 600M v3 VoiceDesign duration config
    ├── train_500m_v3_voice_design_lora.yaml            # 600M v3 VoiceDesign RF+duration LoRA config
    ├── train_500m_v3_lora.yaml               # 500M v3 LoRA fine-tuning config
    ├── train_500m_v3_speaker_inversion.yaml  # 500M v3 Speaker Inversion config
    ├── train_500m_v2.yaml                    # 500M v2 backward-compatible model config
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRA fine-tuning config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign full fine-tuning config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRA fine-tuning config
    ├── train_500m.yaml                       # 500M v1 model config
    └── train_2.5b.yaml                       # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [Irodori-TTS-v4.1-Small model card](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
- [SilentCipher](https://github.com/sony/silentcipher) — Audio watermarking

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuwacom/Irodori-TTS-Optimiz}}
}
```
