import spotipy
from spotipy.oauth2 import SpotifyOAuth
from auth.config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
)

SCOPES = [
    "user-library-read",
    "user-top-read",
    "playlist-read-private",
]

def get_spotify_client():
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_path=".cache/spotify_token"
    )

    return spotipy.Spotify(auth_manager=auth_manager)
