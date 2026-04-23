"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
    _estimate_duration,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


# ---------------------------------------------------------------------------
# Multi-dimensional scorecard
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def dubbing_scorecard(
    metrics:        list[SegmentMetrics],
    aligned:        list[AlignedSegment],
    align_report:   dict | None = None,
) -> dict:
    """Return a normalized multi-dimensional quality score for one clip.

    Dubbing quality cannot be summarised by a single timing number. This
    scorecard breaks it down into four dimensions and returns each on a
    ``[0, 1]`` scale where ``1.0`` = ideal::

        {
            "timing_accuracy":   0.87,   # how well predicted fits source windows
            "stretch_quality":   0.72,   # fraction not requiring severe stretch
            "naturalness":       0.81,   # consistency of speaking rate
            "fit_success":       0.93,   # fraction of segments the policy can serve
            "overall":           0.83,   # weighted mean
            "details": {
                "mean_abs_duration_error_s": ...,
                "pct_severe_stretch":        ...,
                "speaking_rate_std_chars_per_s": ...,
                "n_unfixable":               ...,
            },
        }

    The four dimensions are intentionally designed to capture **independent**
    failure modes so that they do not all move together:

    timing_accuracy
        1.0 when the mean absolute duration error is zero, decays linearly to
        0 at 1.5s error per segment. Measures whether your duration predictor
        is calibrated — i.e. whether ``predicted_tts_s`` matches what TTS
        actually produces. Independent of alignment policy.

    stretch_quality
        Fraction of segments that did **not** need a stretch factor above 1.4
        (the audible distortion threshold for pyrubberband). Measures perceived
        speech quality. A clip can have perfect timing predictions but awful
        stretch quality if translations are too long.

    naturalness
        1.0 when speaking rate (chars/sec) is uniform across segments, decays
        as rate variance grows. A dub that alternates between breakneck and
        glacial delivery reads as unnatural even when individually well-timed.

    fit_success
        Fraction of segments that land in ``ACCEPT``, ``MILD_STRETCH`` or
        ``GAP_SHIFT`` — i.e. the policy can actually produce output. Segments
        hitting ``REQUEST_SHORTER`` or ``FAIL`` count as failures.

    overall
        Weighted mean with weights ``{timing: 0.25, stretch: 0.30,
        naturalness: 0.15, fit: 0.30}`` — stretch and fit matter most because
        they translate directly to audible defects.

    Args:
        metrics: Per-segment timing metrics.
        aligned: Output of ``global_align`` / ``global_align_dp``.
        align_report: Optional pre-computed ``clip_evaluation_report`` dict
            (avoids recomputing if the caller already has it).

    Returns:
        Scorecard dict (see example above).
    """
    if not metrics:
        return {
            "timing_accuracy":  1.0,
            "stretch_quality":  1.0,
            "naturalness":      1.0,
            "fit_success":      1.0,
            "overall":          1.0,
            "details": {
                "mean_abs_duration_error_s":     0.0,
                "pct_severe_stretch":            0.0,
                "speaking_rate_std_chars_per_s": 0.0,
                "n_unfixable":                   0,
            },
        }

    report = align_report or clip_evaluation_report(metrics, aligned)

    # --- timing_accuracy --------------------------------------------------
    mean_err = report["mean_abs_duration_error_s"]
    timing_score = _clip01(1.0 - mean_err / 1.5)

    # --- stretch_quality --------------------------------------------------
    pct_severe = report["pct_severe_stretch"]
    stretch_score = _clip01(1.0 - pct_severe / 100.0)

    # --- naturalness: speaking rate variance -----------------------------
    rates = [
        m.tgt_char_count / max(_estimate_duration(m.translated_text), 0.1)
        for m in metrics if m.tgt_char_count > 0
    ]
    if len(rates) >= 2:
        rate_std = _stats.pstdev(rates)
    else:
        rate_std = 0.0
    # stddev of 0 chars/s = perfect; 5+ chars/s = chaotic.
    naturalness_score = _clip01(1.0 - rate_std / 5.0)

    # --- fit_success ------------------------------------------------------
    n_unfixable = sum(
        1 for a in aligned
        if a.action in (AlignAction.REQUEST_SHORTER, AlignAction.FAIL)
    )
    fit_score = _clip01(1.0 - n_unfixable / max(len(aligned), 1))

    overall = (
        0.25 * timing_score
        + 0.30 * stretch_score
        + 0.15 * naturalness_score
        + 0.30 * fit_score
    )

    return {
        "timing_accuracy":  round(timing_score,      3),
        "stretch_quality":  round(stretch_score,     3),
        "naturalness":      round(naturalness_score, 3),
        "fit_success":      round(fit_score,         3),
        "overall":          round(overall,           3),
        "details": {
            "mean_abs_duration_error_s":     mean_err,
            "pct_severe_stretch":            pct_severe,
            "speaking_rate_std_chars_per_s": round(rate_std, 3),
            "n_unfixable":                   n_unfixable,
        },
    }
