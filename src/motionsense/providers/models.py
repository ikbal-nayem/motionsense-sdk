"""Locating (and, if allowed, fetching) MediaPipe Tasks model bundles.

The legacy ``mediapipe.solutions`` API shipped its models inside the wheel. The
Tasks API that replaced it does not: it takes a ``.task`` bundle path, and
getting that file is the caller's problem. This module makes it not be, while
keeping the network access explicit and opt-out-able -- a library that silently
downloads on import is a bad citizen, but one that fails with "file not found"
and no next step is worse.

Resolution order:

1. an explicit path in :class:`~motionsense.config.EngineConfig`
2. ``$MOTIONSENSE_MODEL_DIR``
3. the per-user cache directory
4. download into the cache, if ``allow_model_download`` is set

Files are written to a temporary name and renamed into place, so an interrupted
download cannot leave a half-written bundle that looks valid.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

__all__ = ["MODELS", "ModelUnavailable", "cache_dir", "resolve_model"]

log = logging.getLogger("motionsense")

_BASE = "https://storage.googleapis.com/mediapipe-models"

#: name -> (filename, download url). Pose complexity maps 0/1/2 to lite/full/heavy.
MODELS: dict[str, tuple[str, str]] = {
    "pose_lite": (
        "pose_landmarker_lite.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    "pose_full": (
        "pose_landmarker_full.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "pose_heavy": (
        "pose_landmarker_heavy.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
    "hand": (
        "hand_landmarker.task",
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}

POSE_BY_COMPLEXITY = {0: "pose_lite", 1: "pose_full", 2: "pose_heavy"}


class ModelUnavailable(RuntimeError):
    """A required model bundle is neither present nor allowed to be fetched."""


def cache_dir() -> Path:
    override = os.environ.get("MOTIONSENSE_MODEL_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "motionsense" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "motionsense" / "models"
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "motionsense" / "models"


def resolve_model(name: str, *, search_dir: str | os.PathLike | None = None, allow_download: bool = True) -> Path:
    """Absolute path to the model bundle ``name``, downloading it if permitted."""
    try:
        filename, url = MODELS[name]
    except KeyError:
        raise ModelUnavailable(f"unknown model {name!r}; known: {sorted(MODELS)}") from None

    candidates = []
    if search_dir:
        candidates.append(Path(search_dir))
    env = os.environ.get("MOTIONSENSE_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    target_dir = cache_dir()
    candidates.append(target_dir)

    for directory in candidates:
        path = Path(directory) / filename
        if path.is_file() and path.stat().st_size > 0:
            return path

    if not allow_download:
        raise ModelUnavailable(
            f"model {filename!r} not found in {[str(c) for c in candidates]}. "
            f"Download it from {url} and place it in one of those directories, "
            f"set MOTIONSENSE_MODEL_DIR, or allow downloads with "
            f"EngineConfig(allow_model_download=True)."
        )

    return _download(url, target_dir / filename)


def _download(url: str, destination: Path) -> Path:
    import urllib.error
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading model %s -> %s", url, destination)

    handle, temporary = tempfile.mkstemp(dir=str(destination.parent), suffix=".part")
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary_path.open("wb") as out:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        if temporary_path.stat().st_size == 0:
            raise ModelUnavailable(f"downloaded an empty file from {url}")
        # Rename, so a partial download is never mistaken for a usable bundle.
        temporary_path.replace(destination)
    except urllib.error.URLError as exc:
        temporary_path.unlink(missing_ok=True)
        raise ModelUnavailable(
            f"could not download {url}: {exc}. Fetch it manually and set "
            f"MOTIONSENSE_MODEL_DIR to the directory containing it."
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
