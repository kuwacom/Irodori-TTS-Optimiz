"""# LongTextSplitter
長文テキストを Irodori-TTS の最大生成時間 (30s) 内に収まるよう、
句読点や記号などの自然な区切りで分割するモジュール

### 特徴
- 句読点・記号位置での自然な分割
- 許可された絵文字 (duration feature 用) を文境界として扱いつつ、分割先文に付与
- 1文が最大文字数を超える場合の強制カット
- 分割後の各セグメントが指定秒数内に収まるよう推定
- duration_scale を考慮: scale > 1 なら話速が遅くなるため、1セグメントに
  詰め込める文字数が減り、より細かく分割される
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .duration import ALLOWED_ANNOTATION_EMOJIS, _ALLOWED_ANNOTATION_EMOJI_PATTERN


# 句読点・記号ベースの分割境界 (これらの直後で分割可能)
_SENTENCE_END_MARKS: tuple[str, ...] = (
    "。",
    "、",
    "，",
    "．",
    "？",
    "！",
    "…",
    "♪",
    "☆",
    "★",
    "～",
    "〜",
    "→",
    "←",
    "」",
    "』",
    "）",
    "】",
    ")",
    ".",
    ",",
    "?",
    "!",
    ";",
    ":",
    "\n",
)

# 句点として扱う文字 (文の終わりとして強い区切り)
_STRONG_END_MARKS: frozenset[str] = frozenset(
    ["。", "？", "！", ".", "?", "!", "」", "』", "）", "）", ")", "\n"]
)

# 分割境界の正規表現パターン (句読点 + 許可絵文字)
_SPLIT_BOUNDARY_CHARS = set(_SENTENCE_END_MARKS) | set(ALLOWED_ANNOTATION_EMOJIS)
_SPLIT_BOUNDARY_PATTERN = re.compile(
    "|".join(
        sorted((re.escape(c) for c in _SPLIT_BOUNDARY_CHARS), key=len, reverse=True)
    )
)


@dataclass(frozen=True)
class SplitSegment:
    """分割後の1セグメント"""
    text: str
    estimatedSeconds: float


@dataclass(frozen=True)
class LongTextSplitResult:
    """長文分割結果"""
    segments: list[SplitSegment]
    wasSplit: bool


def _estimate_seconds_for_text(
    text: str,
    chars_per_second: float,
) -> float:
    """
    テキストから推定生成時間を算出

    日本語TTSでは1文字あたりの平均発話時間をベースに、
    絵文字は制御記号のため短めに見積もる
    """
    # 絵文字を1文字としてカウントしつつ、発話時間への寄与は小さくする
    emoji_hits = _ALLOWED_ANNOTATION_EMOJI_PATTERN.findall(text)
    emoji_count = len(emoji_hits)

    # 発話に寄与する文字数 = 全文字数 - 絵文字数 (絵文字は制御記号として扱い、
    # 発話時間には 0.2 文字分程度の寄与とする)
    char_count = len(text)
    effective_chars = char_count - emoji_count + (emoji_count * 0.2)

    return max(0.5, effective_chars / chars_per_second)


def _find_split_points(text: str) -> list[int]:
    """
    テキスト内の分割候補位置を返す
    返り値は分割境界文字の「直後」のインデックスのリスト
    """
    points: list[int] = []
    for match in _SPLIT_BOUNDARY_PATTERN.finditer(text):
        end_pos = match.end()
        # 境界文字の直後が分割ポイント
        if end_pos < len(text):
            points.append(end_pos)
    return points


def split_long_text(
    text: str,
    *,
    max_seconds: float = 30.0,
    max_chars: int = 180,
    chars_per_second: float = 10.0,
    min_segment_chars: int = 4,
    duration_scale: float = 1.0,
) -> LongTextSplitResult:
    """
    ### split_long_text
    長文テキストを自然な区切りで分割し、各セグメントが
    diffusion最大時間内に収まるようにする

    duration_scale を考慮し、scale > 1 (話速遅い) の場合は
    1セグメントに詰め込める文字数を減らして細かく分割する。
    例: max_seconds=28, duration_scale=2.0 → 実効上限 = 14秒分

    @param text - 分割対象のテキスト
    @param max_seconds - 1セグメントあたりの最大推定秒数 (scale適用前)
    @param max_chars - 1セグメントの最大文字数 (scale適用前の強制カット閾値)
    @param chars_per_second - 1秒あたりの発話文字数の推定 (scale適用前)
    @param min_segment_chars - セグメントの最小文字数 (これ以下なら前セグメントに結合)
    @param duration_scale - 話速スケール (>1 で遅く, <1 で速く)
    @returns 分割結果

    ### 分割ルール
    1. duration_scale で実効上限秒数・文字数を逆算
    2. 句読点・記号の直後で分割 (自然な区切り)
    3. 強い句点 (。？！等) を優先的に分割境界にする
    4. 絵文字は分割境界として扱い、絵文字「後ろ」の次文の先頭には付けない
       (絵文字は直前の文に属する = 発話スタイルの指定)
    5. 1セグメントが実効最大文字数を超える場合、強制カット
    6. 短すぎるセグメントは前のセグメントに結合
    """
    # duration_scale で実効上限を逆算
    # scale=2.0 → 半分の時間しか詰め込めない → 実効上限秒数・文字数も半分
    effective_max_seconds = max_seconds / max(duration_scale, 0.01)
    effective_max_chars = max(1, int(max_chars / max(duration_scale, 0.01)))
    # chars_per_second は scale 前の値 (推定用) なので不変
    # ただし推定秒数が effective_max_seconds を超えないよう判定に使う

    if not text or text.strip() == "":
        return LongTextSplitResult(
            segments=[SplitSegment(text=text or "", estimatedSeconds=0.0)],
            wasSplit=False,
        )

    # 1セグメントで収まる場合は分割不要
    est_seconds = _estimate_seconds_for_text(text, chars_per_second) * duration_scale
    if est_seconds <= max_seconds and len(text) <= effective_max_chars:
        return LongTextSplitResult(
            segments=[SplitSegment(text=text, estimatedSeconds=est_seconds)],
            wasSplit=False,
        )

    # 分割ポイントを取得
    split_points = _find_split_points(text)

    if not split_points:
        # 分割ポイントがない場合は強制カットのみ
        return _force_split(text, max_chars=effective_max_chars, chars_per_second=chars_per_second, duration_scale=duration_scale)

    # 優先分割: 強い句点位置
    strong_points: list[int] = []
    weak_points: list[int] = []
    for sp in split_points:
        # 境界文字 = text[sp-1] (分割位置直前の文字)
        boundary_char = text[sp - 1] if sp > 0 else ""
        if boundary_char in _STRONG_END_MARKS:
            strong_points.append(sp)
        else:
            weak_points.append(sp)

    # 全分割ポイントを優先度順にマージ
    # 強い句点 → 弱い句点の順で走査するため、インデックスでソート
    all_points = sorted(set(strong_points + weak_points))

    # セグメントを構築
    raw_segments: list[str] = []
    prev = 0
    for sp in all_points:
        segment = text[prev:sp].strip()
        if segment:
            raw_segments.append(segment)
        prev = sp

    # 最後の余り
    remainder = text[prev:].strip()
    if remainder:
        if raw_segments:
            raw_segments[-1] += remainder
        else:
            raw_segments.append(remainder)

    # セグメントの結合: 短すぎるセグメントを前のセグメントに統合
    merged_segments: list[str] = []
    for seg in raw_segments:
        if merged_segments and len(seg) < min_segment_chars:
            # 短すぎるセグメントは前のセグメントに結合
            merged_segments[-1] += seg
        else:
            merged_segments.append(seg)

    # 推定秒数・文字数でセグメントを再結合 (隣接セグメントを結合しても
    # effective_max_seconds / effective_max_chars に収まるなら結合する)
    final_segments: list[str] = []
    for seg in merged_segments:
        if not final_segments:
            final_segments.append(seg)
            continue
        combined = final_segments[-1] + seg
        combined_est = _estimate_seconds_for_text(combined, chars_per_second) * duration_scale
        if combined_est <= effective_max_seconds and len(combined) <= effective_max_chars:
            final_segments[-1] = combined
        else:
            final_segments.append(seg)

    # まだ最大文字数を超えるセグメントがあれば強制カット
    result_segments: list[SplitSegment] = []
    for seg in final_segments:
        if len(seg) > effective_max_chars:
            # 強制カットで分割
            force_result = _force_split(
                seg, max_chars=effective_max_chars, chars_per_second=chars_per_second, duration_scale=duration_scale
            )
            result_segments.extend(force_result.segments)
        else:
            est = _estimate_seconds_for_text(seg, chars_per_second) * duration_scale
            result_segments.append(SplitSegment(text=seg, estimatedSeconds=est))

    return LongTextSplitResult(
        segments=result_segments,
        wasSplit=True,
    )


def _force_split(
    text: str,
    *,
    max_chars: int = 200,
    chars_per_second: float = 10.0,
    duration_scale: float = 1.0,
) -> LongTextSplitResult:
    """
    ### _force_split
    分割ポイントがない、または1セグメントが max_chars を超える場合に
    文字数ベースで強制カットする

    @param text - 対象テキスト
    @param max_chars - セグメント最大文字数
    @param chars_per_second - 1秒あたりの発話文字数推定
    @param duration_scale - 話速スケール
    @returns 分割結果
    """
    segments: list[SplitSegment] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            est = _estimate_seconds_for_text(remaining, chars_per_second) * duration_scale
            segments.append(SplitSegment(text=remaining, estimatedSeconds=est))
            break

        # max_chars 位置でカット
        chunk = remaining[:max_chars]
        est = _estimate_seconds_for_text(chunk, chars_per_second) * duration_scale
        segments.append(SplitSegment(text=chunk, estimatedSeconds=est))
        remaining = remaining[max_chars:]

    return LongTextSplitResult(
        segments=segments if segments else [SplitSegment(text=text, estimatedSeconds=0.0)],
        wasSplit=True,
    )
