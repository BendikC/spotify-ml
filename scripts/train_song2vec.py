"""Trains the initial song2vec model from a local copy of the Million
Playlist Dataset (MPD).

Usage:
    python -m scripts.train_song2vec
    python -m scripts.train_song2vec --num-slices 50 --vector-size 200
    python -m scripts.train_song2vec --num-slices 0   # load the full dataset (slow)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from song2vec.corpus import MPD_DIR, find_slice_files, load_corpus
from song2vec.similarity import METADATA_PATH, MODEL_PATH, MODELS_DIR, nearest_neighbors
from song2vec.train import train_word2vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mpd-dir", type=Path, default=MPD_DIR, help="directory containing mpd.slice.*.json files"
    )
    parser.add_argument(
        "--num-slices", type=int, default=10, help="how many slice files to load, in order (0 = all 1,000)"
    )
    parser.add_argument("--min-tracks", type=int, default=2, help="skip playlists shorter than this")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument(
        "--window", type=int, default=None, help="default: whole-playlist context (max playlist length)"
    )
    parser.add_argument("--min-count", type=int, default=5, help="drop tracks appearing in fewer than this many playlists")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not find_slice_files(args.mpd_dir):
        raise SystemExit(
            f"No MPD slice files found in {args.mpd_dir}. Download the Million Playlist "
            "Dataset (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) "
            "and place the mpd.slice.*.json files there."
        )

    sentences, track_metadata = load_corpus(
        args.mpd_dir, num_slices=args.num_slices, min_tracks=args.min_tracks
    )
    if not sentences:
        raise SystemExit(
            f"No playlists with at least {args.min_tracks} tracks were loaded. "
            "Try lowering --min-tracks or pointing at more slices."
        )

    longest = max(len(s) for s in sentences)
    window_desc = args.window if args.window is not None else f"{longest} (whole-playlist context)"
    print(f"Loaded {len(sentences)} playlists, {len(track_metadata)} unique tracks")
    print(f"Longest playlist: {longest} tracks -> using window={window_desc}")

    model = train_word2vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Trained embeddings for {len(model.wv)} / {len(track_metadata)} tracks "
          f"(--min-count={args.min_count} filtered the rest)")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    METADATA_PATH.write_text(json.dumps(track_metadata, indent=2))
    print(f"Saved model to {MODEL_PATH} and metadata to {METADATA_PATH}")

    print("\nSanity check -- nearest neighbors for the most frequent tracks:")
    track_counts = Counter(uri for sentence in sentences for uri in sentence)
    for track_id, _ in track_counts.most_common(5):
        if track_id not in model.wv:
            continue
        meta = track_metadata[track_id]
        print(f"\n{meta['name']} — {', '.join(meta['artists'])}:")
        for neighbor_id, score in nearest_neighbors(model, track_id, topn=5):
            n_meta = track_metadata[neighbor_id]
            print(f"  {score:.3f}  {n_meta['name']} — {', '.join(n_meta['artists'])}")


if __name__ == "__main__":
    main()
