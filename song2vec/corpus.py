"""Loads playlist data from a local copy of the Million Playlist Dataset
(MPD) into song2vec training data: one "sentence" (list of track uris) per
playlist.

Expects one or more `mpd.slice.<start>-<end>.json` files (as distributed by
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
under `data/data/`.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path("data")
MPD_DIR = DATA_DIR / "data"

SLICE_RE = re.compile(r"mpd\.slice\.(\d+)-(\d+)\.json$")


def find_slice_files(mpd_dir=MPD_DIR, num_slices=None):
    """Returns slice file paths sorted by starting playlist id.

    `num_slices` caps how many slices are returned (in order); None/0 means
    all of them. Loading the full dataset (1,000 slices, ~1M playlists) is
    slow, so callers typically want to start with a small subset. A plain
    lexicographic filename sort would misorder slices (e.g. "10000-10999"
    sorts before "2000-2999" as strings), so we sort on the parsed start id.
    """
    files = []
    for path in Path(mpd_dir).glob("mpd.slice.*.json"):
        match = SLICE_RE.search(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda pair: pair[0])
    paths = [path for _, path in files]
    if num_slices:
        paths = paths[:num_slices]
    return paths


def load_corpus(mpd_dir=MPD_DIR, num_slices=10, min_tracks=2):
    """Returns (sentences, track_metadata).

    sentences: list of list[str] track uris, in playlist order, one list
        per playlist. Playlists with fewer than `min_tracks` tracks are
        skipped (MPD playlists have 5-250 tracks by construction, so this
        rarely triggers -- kept for defense in depth). Duplicate track uris
        within a single playlist are left as-is: harmless for word2vec
        training, not worth the extra bookkeeping to dedupe.
    track_metadata: dict track_uri -> {"name": str, "artists": list[str]},
        deduplicated (first-seen) across all loaded playlists. MPD only
        gives a single primary artist per track, so `artists` is always a
        one-element list.
    """
    sentences = []
    track_metadata = {}
    for path in find_slice_files(mpd_dir, num_slices=num_slices):
        slice_data = json.loads(path.read_text())
        for playlist in slice_data["playlists"]:
            tracks = playlist["tracks"]
            uris = [t["track_uri"] for t in tracks]
            if len(uris) < min_tracks:
                continue
            sentences.append(uris)
            for t in tracks:
                track_metadata.setdefault(
                    t["track_uri"],
                    {"name": t["track_name"], "artists": [t["artist_name"]]},
                )
    return sentences, track_metadata
