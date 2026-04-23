"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


_SYLLABLE_RATE = 4.5  # syllables per second for Romance languages

# Per-segment fixed overhead: attack/release silence Chatterbox adds at each segment.
# Calibrated against measured raw_duration_s - syllables/_SYLLABLE_RATE residuals
# on sample clips; tweak if your TTS backend has different acoustic padding.
_TTS_FIXED_OVERHEAD_S = 0.15

# Extra time added per terminal punctuation mark (sentence-final pauses).
_PUNCT_PAUSE_S = 0.18


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds.

    Model::

        duration ≈ syllables / 4.5  +  fixed_overhead  +  punct_pause * terminal_marks

    The syllable term captures the dominant linear relationship with text length.
    The fixed overhead absorbs per-segment attack/release silence that TTS engines
    add regardless of content. Terminal punctuation adds a small additional pause
    because most TTS models render a visible gap at sentence boundaries.

    Against the character-based heuristic (``chars / 15``), this model:

    - correctly penalises polysyllabic words (``internacionalización`` scores as a
      long utterance even though it is one token),
    - avoids underestimating very short segments (single-word replies still have
      the fixed overhead),
    - responds to sentence-terminal pauses that stretch speech beyond what
      syllable count alone predicts.

    The formula is intentionally stdlib-only so that no ML dependency is needed
    for pure-Python alignment work. A learned regression model can replace this
    when ground-truth WAV durations are available — the call site stays the same.
    """
    text = text.strip()
    if not text:
        return 0.0
    n_syllables = _count_syllables(text)
    n_terminal = sum(text.count(p) for p in (".", "!", "?"))
    return (
        n_syllables / _SYLLABLE_RATE
        + _TTS_FIXED_OVERHEAD_S
        + _PUNCT_PAUSE_S * n_terminal
    )


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


# ---------------------------------------------------------------------------
# Global DP-based optimizer
# ---------------------------------------------------------------------------

# Per-action penalties (cost function). Lower = preferred.
# Calibrated so the optimizer reaches for gap_shift *before* stretching audibly,
# but still prefers ACCEPT over any modification and treats FAIL as a last resort.
_DP_PENALTY = {
    AlignAction.ACCEPT:          0.0,
    AlignAction.MILD_STRETCH:    1.0,
    AlignAction.GAP_SHIFT:       0.5,
    AlignAction.REQUEST_SHORTER: 4.0,
    AlignAction.FAIL:            20.0,
}

# Weight on accumulated drift in seconds. Tunes how strongly the optimizer
# resists drifting the downstream timeline vs taking a local penalty now.
_DP_DRIFT_WEIGHT = 0.8


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    drift_weight:    float = _DP_DRIFT_WEIGHT,
) -> list[AlignedSegment]:
    """Dynamic-programming global alignment — a better optimizer than ``global_align``.

    **Why this exists**

    ``global_align`` is greedy: it commits to an action for each segment without
    looking ahead. That causes two classes of failure:

    1. **Wasted silence** — segment *k* accepts a mild stretch when there is a
       large silence gap right after it, then segment *k+1* fails because it
       needed that silence.
    2. **Cumulative drift** — each gap shift pushes the entire tail of the clip
       forward. The greedy pass never balances *local* penalty against the
       *downstream* cost of the drift it introduces.

    **Approach**

    Standard DP over segments. For each segment *i* we enumerate the feasible
    actions and, for each, record a per-action cost plus a drift penalty
    proportional to the cumulative drift after applying the action. The DP
    minimises the sum of (action penalty + drift penalty) end-to-end:

    .. math::

        C[i][\\text{drift}] = \\min_a \\; \\big( p(a) + w \\cdot \\text{drift}_\\text{new} + C[i+1][\\text{drift}_\\text{new}] \\big)

    Because cumulative drift is a continuous variable we discretise it into
    100-ms buckets — plenty of resolution for dubbing timing and bounded enough
    to make the DP O(n · buckets · actions).

    **Differences from the greedy pass**

    - Considers ``GAP_SHIFT`` even when the local stretch would fit, so silence
      is reserved for a later segment that will need it more.
    - Balances ``MILD_STRETCH`` against cumulative drift instead of always
      preferring whichever action happens to be locally cheapest.
    - Still respects the same fixed thresholds in ``decide_action`` to stay
      consistent with downstream policy / reporting.

    Args:
        metrics: Per-segment timing metrics.
        silence_regions: VAD silence list (same shape as ``global_align``).
        max_stretch: Upper bound on the ``MILD_STRETCH`` factor.
        drift_weight: Multiplier on cumulative drift (seconds). Higher values
            make the optimizer more conservative about borrowing silence.

    Returns:
        One ``AlignedSegment`` per metric, in order, with the schedule chosen
        by the DP.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    n = len(metrics)
    if n == 0:
        return []

    # Pre-compute feasible action candidates per segment.
    # Each candidate is (action, gap_shift_s, stretch_factor, local_penalty).
    per_seg_candidates: list[list[tuple[AlignAction, float, float, float]]] = []
    for m in metrics:
        gap = _silence_after(m.source_end)
        greedy_choice = decide_action(m, available_gap_s=gap)

        cands: list[tuple[AlignAction, float, float, float]] = []

        # ACCEPT is always available — may mean the audio is clipped, but the
        # DP needs the option. Penalise when it would require stretching above
        # the mild-stretch threshold to stay in window.
        if m.predicted_stretch <= 1.1:
            cands.append((AlignAction.ACCEPT, 0.0, 1.0, _DP_PENALTY[AlignAction.ACCEPT]))

        if m.predicted_stretch <= max_stretch + 0.4:
            stretch = min(max(m.predicted_stretch, 1.0), max_stretch)
            # Scale penalty by how aggressive the stretch is (0 at 1.0, 1.0 at max_stretch).
            stretch_frac = (stretch - 1.0) / max(max_stretch - 1.0, 1e-6)
            cands.append((
                AlignAction.MILD_STRETCH, 0.0, stretch,
                _DP_PENALTY[AlignAction.MILD_STRETCH] * max(0.1, stretch_frac),
            ))

        if gap >= m.overflow_s > 0.0:
            cands.append((
                AlignAction.GAP_SHIFT, m.overflow_s, 1.0,
                _DP_PENALTY[AlignAction.GAP_SHIFT],
            ))

        if greedy_choice == AlignAction.REQUEST_SHORTER:
            cands.append((AlignAction.REQUEST_SHORTER, 0.0, 1.0, _DP_PENALTY[AlignAction.REQUEST_SHORTER]))

        # Always provide FAIL as a last-resort fallback so the DP never has an empty set.
        cands.append((AlignAction.FAIL, 0.0, 1.0, _DP_PENALTY[AlignAction.FAIL]))

        per_seg_candidates.append(cands)

    # Discretise cumulative drift into 100-ms buckets and cap it so the state
    # space stays small. Any sane dubbing clip keeps |drift| < 10s.
    BUCKET = 0.1
    MAX_DRIFT_S = 10.0
    MAX_BUCKETS = int(MAX_DRIFT_S / BUCKET) + 1

    # DP: best[i][drift_bucket] -> (total_cost, chosen_candidate_idx, next_bucket)
    # Solve backwards from i=n-1 to 0.
    INF = float("inf")
    best: list[dict[int, tuple[float, int, int]]] = [dict() for _ in range(n + 1)]
    best[n][0] = (0.0, -1, 0)  # terminal state

    for i in range(n - 1, -1, -1):
        for prev_bucket in range(MAX_BUCKETS):
            drift_s = prev_bucket * BUCKET
            best_cost = INF
            best_tuple: tuple[float, int, int] | None = None
            for idx, (_action, gap_shift, _stretch, pen) in enumerate(per_seg_candidates[i]):
                new_drift = min(MAX_DRIFT_S, drift_s + gap_shift)
                new_bucket = int(round(new_drift / BUCKET))
                future = best[i + 1].get(new_bucket)
                if future is None:
                    continue
                cost = pen + drift_weight * gap_shift + future[0]
                if cost < best_cost:
                    best_cost = cost
                    best_tuple = (cost, idx, new_bucket)
            if best_tuple is not None:
                best[i][prev_bucket] = best_tuple

    # Reconstruct
    aligned: list[AlignedSegment] = []
    bucket = 0
    cumulative_drift = 0.0
    for i in range(n):
        entry = best[i].get(bucket)
        if entry is None:
            # Fallback to greedy if DP left a hole (shouldn't happen).
            return global_align(metrics, silence_regions, max_stretch)
        _cost, idx, next_bucket = entry
        action, gap_shift, stretch, _pen = per_seg_candidates[i][idx]
        m = metrics[i]
        sched_start = m.source_start + cumulative_drift
        sched_end = sched_start + m.source_duration_s + gap_shift
        aligned.append(AlignedSegment(
            index=m.index,
            original_start=m.source_start,
            original_end=m.source_end,
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            text=m.translated_text,
            action=action,
            gap_shift_s=gap_shift,
            stretch_factor=stretch,
        ))
        cumulative_drift += gap_shift
        bucket = next_bucket

    return aligned
