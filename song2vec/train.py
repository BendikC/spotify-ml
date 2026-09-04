"""Trains word2vec-style track embeddings: treating each playlist as a
"sentence" of track ids, tracks that co-occur across playlists end up close
together in embedding space (the same idea as prod2vec/playlist2vec).

Uses a "whole playlist" context window rather than a small sequential one:
playlist ordering is often mood/curation-driven rather than a meaningful
adjacency signal, so every track in a playlist should be context for every
other track regardless of position.

Note: training with workers > 1 is not bit-for-bit reproducible across runs
even with a fixed seed (gensim's multi-threaded SGD has nondeterministic
update ordering). That's an acceptable tradeoff for training speed here.
"""

from gensim.models import Word2Vec


def train_word2vec(
    sentences,
    vector_size=100,
    window=None,
    min_count=5,
    epochs=10,
    sg=1,
    workers=4,
    seed=42,
):
    if window is None:
        window = max(len(sentence) for sentence in sentences)
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        # Without this, gensim samples a random effective window <= `window`
        # per training instance (its default dynamic-window trick), which
        # reintroduces a distance bias -- defeating the whole-playlist,
        # order-agnostic context this project wants.
        shrink_windows=False,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
