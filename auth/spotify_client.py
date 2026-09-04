from auth.auth import get_spotify_client

class SpotifyClient:
    def __init__(self):
        self.sp = get_spotify_client()

    def get_top_tracks(self, limit=20, time_range="medium_term"):
        return self.sp.current_user_top_tracks(
            limit=limit, time_range=time_range
        )

    def get_top_artists(self, limit=20, time_range="medium_term"):
        return self.sp.current_user_top_artists(
            limit=limit, time_range=time_range
        )

    def search_playlists(self, query, limit=50, offset=0):
        return self.sp.search(
            q=query, type="playlist", limit=limit, offset=offset
        )

    def get_playlist_tracks(self, playlist_id):
        """Yields playlist item dicts, paginating through the full playlist."""
        results = self.sp.playlist_items(
            playlist_id, additional_types=["track"], limit=100
        )
        while results:
            for item in results["items"]:
                yield item
            results = self.sp.next(results) if results.get("next") else None

    def get_audio_features(self, track_ids):
        return self.sp.audio_features(track_ids)
