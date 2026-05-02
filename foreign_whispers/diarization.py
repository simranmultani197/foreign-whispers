"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


# torchaudio 2.7+ removed several module-level APIs that pyannote.audio 3.4.0
# still references at import time. Restore stubs so the imports resolve — the
# real work is done via APIs that still exist (torchaudio.info, soundfile).
# Run at module load so the shim is in place before anything (tests, services,
# notebooks) imports pyannote.
try:
    import torchaudio as _torchaudio

    # AudioMetaData: used as a return-type annotation in pyannote/audio/core/io.py.
    if not hasattr(_torchaudio, "AudioMetaData"):
        class _AudioMetaDataShim:
            """Stub for pyannote.audio compatibility on torchaudio>=2.7."""
        _torchaudio.AudioMetaData = _AudioMetaDataShim

    # list_audio_backends: removed in 2.7 along with the legacy backend system.
    # pyannote.audio.core.io.Audio.__init__ calls it for validation only;
    # returning a non-empty list is enough to satisfy the check.
    if not hasattr(_torchaudio, "list_audio_backends"):
        _torchaudio.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

# PyTorch 2.6 flipped torch.load's default to weights_only=True. pyannote.audio's
# checkpoints contain classes (TorchVersion, omegaconf, lightning) that aren't on
# the default safe-globals allowlist, so pyannote's model load fails with
# `WeightsUnpickler error: Unsupported global`. The pyannote weights come from
# the gated pyannote/* HF repos signed by the pyannote team, so weights_only=False
# is an acceptable tradeoff here. Patch torch.load to default to False unless the
# caller explicitly opts in.
try:
    import torch as _torch
    if not getattr(_torch.load, "_fw_patched", False):
        _orig_torch_load = _torch.load

        def _patched_torch_load(*args, **kwargs):
            # Force, don't setdefault — pytorch-lightning's load_from_checkpoint
            # passes weights_only=True explicitly on newer versions, which would
            # bypass setdefault and re-trigger the WeightsUnpickler error.
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        _patched_torch_load._fw_patched = True
        _torch.load = _patched_torch_load
except ImportError:
    pass


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError, AttributeError) as exc:
        logger.warning("pyannote.audio import failed (%s) — returning empty diarization.", exc)
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    labeled: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for turn in diarization:
            overlap = max(
                0.0,
                min(seg_end, turn["end_s"]) - max(seg_start, turn["start_s"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        new_seg["speaker"] = best_speaker
        labeled.append(new_seg)
    return labeled
