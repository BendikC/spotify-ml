from auth.auth import get_spotify_client

class SpotifyClient:
    def __init__(self):
        self.sp = get_spotify_client()

    def get_top_tracks(self, limit=20, time_range="medium_term"):
        return self.sp.current_user_top_tracks(
            limit=limit, time_range=time_range
        )

    def get_audio_features(self, track_ids):
        return self.sp.audio_features(track_ids)
