"""Looks up a track by (partial, case-insensitive) name and prints its
nearest neighbors in the trained song2vec embedding space.

Usage:
    python -m scripts.find_similar "bohemian rhapsody"
    python -m scripts.find_similar "yesterday" --topn 20
"""

import argparse
import json

from gensim.models import Word2Vec

from song2vec.similarity import METADATA_PATH, MODEL_PATH, find_track, nearest_neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="substring to match against track names")
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise SystemExit(
            f"No trained model found at {MODEL_PATH}. Run 'python -m scripts.train_song2vec' first."
        )

    model = Word2Vec.load(str(MODEL_PATH))
    track_metadata = json.loads(METADATA_PATH.read_text())

    track_id, meta, match_count = find_track(track_metadata, args.name)
    if track_id is None:
        raise SystemExit(f"No track matching {args.name!r} found.")
    if match_count > 1:
        print(
            f"{match_count} tracks matched {args.name!r}; showing neighbors for: "
            f"{meta['name']} — {', '.join(meta['artists'])}"
        )

    if track_id not in model.wv:
        raise SystemExit(
            f"{meta['name']!r} appeared in the corpus but was filtered out by "
            "--min-count during training (too rare). Retrain with a lower "
            "--min-count to include it."
        )

    print(f"Nearest neighbors to {meta['name']!r} by {', '.join(meta['artists'])}:")
    for i, (neighbor_id, score) in enumerate(nearest_neighbors(model, track_id, topn=args.topn), 1):
        n_meta = track_metadata[neighbor_id]
        print(f"  {i}. {n_meta['name']} — {', '.join(n_meta['artists'])} ({score:.3f})")


if __name__ == "__main__":
    main()
