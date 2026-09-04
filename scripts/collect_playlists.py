"""
Builds a local corpus of playlists (and their tracklists) for the song2vec
project: search playlists seeded from your own top-artist genres, then fetch
each playlist's tracks.

Usage:
    python -m scripts.collect_playlists
    python -m scripts.collect_playlists --pages-per-query 4 --max-playlists 500
    python -m scripts.collect_playlists --skip-search   # reuse data/playlists.json,
                                                          # just (re)fetch tracks
"""

import argparse
import json
import sys
from pathlib import Path

# Windows consoles often default to cp1252, which can't print many artist
# names; keep output printable instead of crashing.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spotipy.exceptions import SpotifyException

from auth.spotify_client import SpotifyClient

DATA_DIR = Path("data")
PLAYLISTS_FILE = DATA_DIR / "playlists.json"
TRACKS_DIR = DATA_DIR / "playlist_tracks"

# Generic terms mixed in alongside your genres/artists, so the corpus isn't
# limited to playlists that only ever use genre names in their titles.
DEFAULT_SEED_TERMS = ["chill", "workout", "focus", "party", "road trip", "sad", "throwback"]

# Spotify's documented max for search `limit` is 50, but for this app the
# endpoint rejects anything above 10 with a 400 "Invalid limit" (verified
# empirically 2026-07; undocumented dev-mode restriction). Results also
# contain null entries, and pagination stops after ~3 pages (~30 items),
# so expect roughly 10-15 real playlists per query.
DEFAULT_PAGE_SIZE = 10


def collect_seed_queries(client, limit=15):
    genres = set()
    artist_names = set()
    for time_range in ("short_term", "medium_term", "long_term"):
        artists = client.get_top_artists(limit=limit, time_range=time_range)
        for artist in artists["items"]:
            genres.update(artist.get("genres", []))
            artist_names.add(artist["name"])

    genres = sorted(genres)
    if len(genres) < 5:
        # Spotify's artist `genres` field is sparsely populated for most
        # artists these days, so genres alone often isn't enough signal.
        print(
            f"Only {len(genres)} genre(s) found on your top artists "
            "(Spotify's artist-genre data is often empty) — "
            "adding your top artist names as search seeds too."
        )
        return genres + sorted(artist_names)
    return genres


def search_playlists_for_query(client, query, pages=2, page_size=DEFAULT_PAGE_SIZE):
    playlists = {}
    for page in range(pages):
        offset = page * page_size
        try:
            results = client.search_playlists(query, limit=page_size, offset=offset)
        except SpotifyException as e:
            if e.http_status == 400:
                print(f"  search rejected limit={page_size} for {query!r}, stopping this query")
                break
            raise
        block = results["playlists"]
        items = [item for item in block["items"] if item is not None]
        if not items:
            break
        for item in items:
            # dev-mode search responses omit some documented fields
            playlists[item["id"]] = {
                "id": item["id"],
                "name": item.get("name", ""),
                "owner": (item.get("owner") or {}).get("display_name", ""),
                "tracks_total": (item.get("tracks") or {}).get("total"),
                "query": query,
            }
        if block["next"] is None:
            break
    return playlists


def collect_playlists(client, queries, pages_per_query=2):
    all_playlists = {}
    for query in queries:
        print(f"Searching playlists for: {query!r}")
        found = search_playlists_for_query(client, query, pages=pages_per_query)
        print(f"  found {len(found)} playlists")
        all_playlists.update(found)
    return all_playlists


def fetch_playlist_tracks(client, playlist_id):
    tracks = []
    for item in client.get_playlist_tracks(playlist_id):
        track = item.get("track")
        if not track or not track.get("id"):
            continue  # local files / removed tracks have no id
        tracks.append(
            {
                "id": track["id"],
                "name": track["name"],
                "artists": [a["name"] for a in track["artists"]],
                "artist_ids": [a["id"] for a in track["artists"]],
            }
        )
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-per-query", type=int, default=3, help=f"{DEFAULT_PAGE_SIZE} results per page; search stops serving results after ~3 pages")
    parser.add_argument("--max-playlists", type=int, default=300, help="cap on playlists to fetch tracks for")
    parser.add_argument("--skip-search", action="store_true", help="reuse existing data/playlists.json")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    TRACKS_DIR.mkdir(exist_ok=True, parents=True)

    client = SpotifyClient()

    if args.skip_search and PLAYLISTS_FILE.exists():
        all_playlists = json.loads(PLAYLISTS_FILE.read_text())
    else:
        seed_queries = collect_seed_queries(client)
        print(f"Seed queries from your listening history ({len(seed_queries)}): {seed_queries}")
        queries = seed_queries + DEFAULT_SEED_TERMS
        all_playlists = collect_playlists(client, queries, pages_per_query=args.pages_per_query)
        PLAYLISTS_FILE.write_text(json.dumps(all_playlists, indent=2))
        print(f"Saved {len(all_playlists)} unique playlists to {PLAYLISTS_FILE}")

    playlist_ids = list(all_playlists.keys())[: args.max_playlists]
    print(f"Fetching tracks for {len(playlist_ids)} playlists...")

    for i, playlist_id in enumerate(playlist_ids, 1):
        out_file = TRACKS_DIR / f"{playlist_id}.json"
        if out_file.exists():
            continue  # resumable: skip playlists already fetched
        try:
            tracks = fetch_playlist_tracks(client, playlist_id)
        except Exception as e:
            print(f"  [{i}/{len(playlist_ids)}] failed {playlist_id}: {e}")
            continue
        out_file.write_text(json.dumps(tracks, indent=2))
        name = all_playlists[playlist_id]["name"]
        print(f"  [{i}/{len(playlist_ids)}] {name!r}: {len(tracks)} tracks")

    print("Done.")


if __name__ == "__main__":
    main()
