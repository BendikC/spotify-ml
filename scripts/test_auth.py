from auth.spotify_client import SpotifyClient

def main():
    client = SpotifyClient()
    results = client.get_top_tracks(limit=5)

    for track in results["items"]:
        print(
            f"{track['name']} – {track['artists'][0]['name']}"
        )

if __name__ == "__main__":
    main()
