import pandas as pd

tracksData = pd.read_csv('spotify-tracks-dataset.csv')

# all the columns are
# ['track_id', 'artists', 'album_name', 'track_name', 
# 'popularity', 'duration_ms', 'explicit', 'danceability', 
# 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
# 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre']

def clean_data(data):
    data = data.dropna()

    # Remove unnecessary columns
    data = data.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'key', 'mode', 'time_signature', 'track_genre'])
    data = data.drop(data.columns[0], axis=1)

    # add a column for the danceability level
    data['danceability_level'] = pd.cut(data['danceability'], bins=[0.0, 0.33, 0.66, 1.0], labels=[0, 1, 2])

    # re-arrange the columns
    data = data[['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'danceability', 'danceability_level']]

    # make a new csv file with the cleaned data
    data.to_csv('cleaned_spotify_tracks_data.csv', index=False)

clean_data(tracksData)