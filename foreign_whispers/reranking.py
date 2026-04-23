"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    CHARS_PER_SECOND = 15.0
    budget_chars = max(1, int(round(target_duration_s * CHARS_PER_SECOND)))

    candidates: list[TranslationCandidate] = []
    seen: set[str] = set()

    def _push(text: str, rationale: str) -> None:
        text = " ".join(text.split()).strip()
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(TranslationCandidate(
            text=text,
            char_count=len(text),
            brevity_rationale=rationale,
        ))

    _push(baseline_es, "baseline (unchanged)")

    shortened = _apply_phrase_contractions(baseline_es)
    if shortened != baseline_es:
        _push(shortened, "phrase contractions applied")

    stripped = _strip_filler_words(shortened)
    if stripped != shortened:
        _push(stripped, "filler words removed")

    synonymized = _apply_synonym_shortening(stripped)
    if synonymized != stripped:
        _push(synonymized, "shorter synonyms substituted")

    if len(synonymized) > budget_chars:
        truncated = _truncate_to_clause(synonymized, budget_chars)
        if truncated and truncated != synonymized:
            _push(truncated, f"truncated at clause boundary to ~{budget_chars} chars")

    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: %d candidates for %.1fs budget (~%d chars), "
        "shortest=%d chars, baseline=%d chars",
        len(candidates),
        target_duration_s,
        budget_chars,
        candidates[0].char_count if candidates else 0,
        len(baseline_es),
    )
    return candidates


_PHRASE_CONTRACTIONS: list[tuple[str, str]] = [
    ("en este momento", "ahora"),
    ("en estos momentos", "ahora"),
    ("en el día de hoy", "hoy"),
    ("en el día de ayer", "ayer"),
    ("a pesar de que", "aunque"),
    ("a pesar de", "pese a"),
    ("con el fin de", "para"),
    ("con el objeto de", "para"),
    ("con el objetivo de", "para"),
    ("con la finalidad de", "para"),
    ("debido a que", "porque"),
    ("puesto que", "porque"),
    ("ya que", "porque"),
    ("dado que", "porque"),
    ("de manera que", "así que"),
    ("de modo que", "así que"),
    ("de tal manera que", "así que"),
    ("en relación con", "sobre"),
    ("con respecto a", "sobre"),
    ("en cuanto a", "sobre"),
    ("a fin de que", "para que"),
    ("hacer referencia a", "referirse a"),
    ("tener en cuenta", "considerar"),
    ("llevar a cabo", "realizar"),
    ("dar comienzo a", "iniciar"),
    ("poner de manifiesto", "mostrar"),
    ("un gran número de", "muchos"),
    ("la mayor parte de", "la mayoría de"),
    ("una gran cantidad de", "muchos"),
    ("de forma que", "así que"),
    ("así como también", "y"),
    ("así como", "y"),
    ("sin embargo", "pero"),
    ("no obstante", "pero"),
]


_FILLER_WORDS: set[str] = {
    "pues", "bueno", "básicamente", "obviamente", "evidentemente",
    "efectivamente", "realmente", "precisamente", "justamente",
    "prácticamente", "literalmente", "simplemente", "exactamente",
    "verdaderamente", "ciertamente", "quizás", "tal vez",
}


_SYNONYMS: list[tuple[str, str]] = [
    ("utilizar", "usar"),
    ("utilización", "uso"),
    ("solicitar", "pedir"),
    ("necesitar", "requerir"),  # both 8, skip? keep consistent direction
    ("comenzar", "iniciar"),
    ("finalizar", "terminar"),
    ("obtener", "lograr"),
    ("realizar", "hacer"),
    ("efectuar", "hacer"),
    ("observar", "ver"),
    ("visualizar", "ver"),
    ("demostrar", "probar"),
    ("manifestar", "decir"),
    ("comunicar", "decir"),
    ("expresar", "decir"),
    ("adquirir", "comprar"),
    ("responder", "contestar"),  # longer, drop direction
    ("aproximadamente", "unos"),
    ("anteriormente", "antes"),
    ("posteriormente", "luego"),
    ("actualmente", "hoy"),
    ("frecuentemente", "a menudo"),
    ("habitualmente", "suele"),
    ("muy grande", "enorme"),
    ("muy pequeño", "mínimo"),
    ("muy importante", "clave"),
]


def _apply_phrase_contractions(text: str) -> str:
    """Replace wordy Spanish phrases with shorter equivalents (case-preserving start)."""
    import re as _re
    result = text
    for long_phrase, short in _PHRASE_CONTRACTIONS:
        pattern = _re.compile(r"\b" + _re.escape(long_phrase) + r"\b", _re.IGNORECASE)
        result = pattern.sub(short, result)
    return result


def _strip_filler_words(text: str) -> str:
    """Remove common Spanish filler adverbs when they carry no semantic weight."""
    import re as _re
    tokens = _re.split(r"(\s+|[,;.!?])", text)
    kept = [t for t in tokens if t.strip().lower() not in _FILLER_WORDS]
    out = "".join(kept)
    out = _re.sub(r"\s+([,;.!?])", r"\1", out)
    out = _re.sub(r",\s*,", ",", out)
    return _re.sub(r"\s+", " ", out).strip()


def _apply_synonym_shortening(text: str) -> str:
    """Replace longer Spanish words with shorter near-synonyms when char_count drops."""
    import re as _re
    result = text
    for long_word, short in _SYNONYMS:
        if len(short) >= len(long_word):
            continue
        pattern = _re.compile(r"\b" + _re.escape(long_word) + r"\b", _re.IGNORECASE)
        result = pattern.sub(short, result)
    return result


def _truncate_to_clause(text: str, budget_chars: int) -> str:
    """Truncate *text* at the last clause boundary (comma/semicolon/period) within budget.

    Adds a trailing ellipsis if truncation happened mid-sentence. Returns the
    original string when no safe cut point is available.
    """
    if len(text) <= budget_chars:
        return text

    window = text[:budget_chars]
    for sep in (". ", "; ", ", "):
        cut = window.rfind(sep)
        if cut >= budget_chars * 0.4:
            return window[: cut + 1].strip()

    cut = window.rfind(" ")
    if cut >= budget_chars * 0.5:
        return window[:cut].rstrip(",;. ") + "…"

    return text
