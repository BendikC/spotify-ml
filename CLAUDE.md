# spotify-ml / song2vec

Personal project training track embeddings ("song2vec") on playlist
co-occurrence data, using a local copy of the **Million Playlist Dataset
(MPD)**.

## Data source

The corpus comes from the MPD (not the Spotify Web API — an earlier
approach using `scripts/collect_playlists.py` + a Spotify OAuth client was
abandoned because API access was too limited, and that code has been
removed). The dataset is 1,000 JSON slice files
(`mpd.slice.<start>-<end>.json`, ~1,000 playlists each), downloaded from
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
and extracted into `data/data/` (gitignored — the dataset itself is never
committed).

Each track in a playlist carries `track_uri`, `track_name`, `artist_name`,
`album_name`, `duration_ms`, and `pos`. MPD only gives one primary artist
per track.

## File layout

- `song2vec/` — the library:
  - `corpus.py` — `find_slice_files()` / `load_corpus()`: parses MPD slices
    into `sentences` (one `list[str]` of track uris per playlist, in `pos`
    order) and `track_metadata` (`track_uri -> {"name", "artists"}`).
  - `train.py` — `train_word2vec()`: thin wrapper around
    `gensim.models.Word2Vec`.
  - `similarity.py` — shared model/metadata paths (`models/song2vec.model`,
    `models/track_metadata.json`) and lookup helpers (`find_track()`,
    `nearest_neighbors()`) used by both the training sanity check and the
    query script.
- `scripts/` — thin CLIs:
  - `train_song2vec.py` — loads the corpus, trains, saves the model, prints
    a sanity check.
  - `find_similar.py` — looks up a track by name and prints its neighbors.

`models/` (like `data/`) is gitignored — trained artifacts are
regeneratable, not checked in.

## Design choices

- **Whole-playlist context window**: `train_word2vec()` defaults
  `window=None`, which is resolved internally to
  `max(len(s) for s in sentences)` — effectively "every track in a playlist
  is context for every other track," regardless of position. This is a
  deliberate choice: playlist ordering is often mood/curation-driven rather
  than a meaningful "flow" signal, so a small sequential window would encode
  a bias that isn't really there. gensim clamps the per-token effective
  window to each sentence's actual length, so one large global `window`
  value is equivalent to "whole playlist" for every sentence regardless of
  length.
- **`shrink_windows=False`** (gensim ≥4.1.0): required for the whole-playlist
  design to actually hold. gensim's default (`shrink_windows=True`) samples
  a random smaller effective window per training instance (a dynamic-window
  trick that weights nearby words more heavily) — leaving it on would
  silently reintroduce the exact distance bias the whole-playlist window is
  meant to avoid.
- **`min_count=5`** default: MPD-scale corpora (even the default
  `--num-slices 10`, ~10k playlists) are large enough that this is a
  reasonable floor to keep the vocabulary to tracks with real co-occurrence
  signal. It's CLI-overridable — lower it for small `--num-slices` runs.
- Duplicate track uris within one playlist, and playlists under
  `--min-tracks` (default 2), are left unhandled/skipped respectively; see
  the docstrings in `corpus.py` for why each is fine as-is.
- Training uses `workers=4` by default, so results aren't bit-for-bit
  reproducible across runs even with a fixed seed (multi-threaded SGD has
  nondeterministic update ordering) — acceptable for training speed on a
  personal project.

## Usage

```bash
pip install -r requirements.txt

# place MPD slice files under data/data/, then:
python -m scripts.train_song2vec
python -m scripts.train_song2vec --num-slices 50 --vector-size 200
python -m scripts.find_similar "song name"
```
