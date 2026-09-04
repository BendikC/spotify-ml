# Spotify ML

Personal project training track embeddings (song2vec) on playlist
co-occurrence data from the Million Playlist Dataset.

## Setup

1. Download the Million Playlist Dataset:
   https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
2. Extract the `mpd.slice.*.json` files into `data/data/`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python -m scripts.train_song2vec
python -m scripts.find_similar "song name"
```
