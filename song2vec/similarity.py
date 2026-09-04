"""Shared lookup helpers for querying a trained song2vec model by track
name, used by both the post-training sanity check and scripts/find_similar.py.
"""

from pathlib import Path

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "song2vec.model"
METADATA_PATH = MODELS_DIR / "track_metadata.json"


def find_track(track_metadata, name_substring):
    """Case-insensitive substring match against track names.

    Returns (track_id, meta, match_count); (None, None, 0) if nothing
    matches. Ties broken by track_metadata's (deterministic, first-seen)
    order.
    """
    query = name_substring.lower()
    matches = [
        (tid, meta) for tid, meta in track_metadata.items() if query in meta["name"].lower()
    ]
    if not matches:
        return None, None, 0
    track_id, meta = matches[0]
    return track_id, meta, len(matches)


def nearest_neighbors(model, track_id, topn=10):
    return model.wv.most_similar(track_id, topn=topn)
