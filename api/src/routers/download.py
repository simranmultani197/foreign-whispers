"""POST /api/download — download YouTube video + captions (issue by5)."""

import json
import pathlib
import re

from fastapi import APIRouter, HTTPException, Request

from api.src.core.config import settings
from api.src.core.video_registry import get_video
from api.src.schemas.download import CaptionSegment, DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


def _extract_video_id_from_url(url: str) -> str | None:
    """Extract 11-char YouTube video ID from URL without hitting the network."""
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments.

    Skips the YouTube round-trip entirely when the registry knows the title
    AND both files already exist on disk — useful when videos were placed
    manually (e.g. side-loaded around geo-restrictions).
    """
    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Try to short-circuit the YouTube call if everything is already cached.
    quick_id = _extract_video_id_from_url(body.url)
    quick_entry = get_video(quick_id) if quick_id else None
    if quick_entry:
        quick_video = videos_dir / f"{quick_entry.title}.mp4"
        quick_caption = captions_dir / f"{quick_entry.title}.txt"
        if quick_video.exists() and quick_caption.exists():
            segments = _download_service.read_caption_segments(quick_caption)
            return DownloadResponse(
                video_id=quick_id,
                title=quick_entry.title,
                caption_segments=segments,
            )

    # At least one artifact is missing — go to YouTube for canonical title.
    try:
        video_id, title = _download_service.get_video_info(body.url)
    except Exception as exc:
        # If we have a registry entry and the video file already exists,
        # we can still continue without captions (use registry title).
        if quick_entry and quick_id:
            quick_video = videos_dir / f"{quick_entry.title}.mp4"
            if quick_video.exists():
                return DownloadResponse(
                    video_id=quick_id,
                    title=quick_entry.title,
                    caption_segments=[],
                )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Use title from registry; fall back to yt-dlp title with colons stripped
    entry = get_video(video_id)
    stem = entry.title if entry else title.replace(":", "")

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    # Skip re-download if both files exist
    if not video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), stem)

    if not caption_path.exists():
        _download_service.download_caption(body.url, str(captions_dir), stem)

    segments = _download_service.read_caption_segments(caption_path)

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )
