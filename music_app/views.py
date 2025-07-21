from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo'] #features of the data



sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='054ae02eced440f59261535505c8f516', client_secret='e8978ae2ca60408cbaa2a33aecc3271f'))


def load_csv(file_name): #method to load csv files
    file_path = os.path.join(settings.BASE_DIR, 'music_app', 'data_files', file_name)
    return pd.read_csv(file_path)

def find_song(name, year): #searches for song by name and year and returns features in df. 
    song_data = defaultdict() #empty dict that will store song info
    results = sp.search(q= 'track: {} year: {}'.format(name, year), limit=1) #sp is spotify client, searches from client where q is query in the format. limit ensures just one result is returned
    if results['tracks']['items'] == []: #if no tracks in search result, returns none
        return None

    results = results['tracks']['items'][0] #if track found, extracts only first track from res and gets its id
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0] #gets audio features of the first track
    #store data in dict song_data as a list: becomes a dict containing lists
    song_data['name'] = [name] 
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items(): #stores value in dict song_data under corresponding key for each key-value pair in audio_features 
        song_data[key] = value

    return pd.DataFrame(song_data) #converts dict to df

def get_song_data(song, spotify_data):
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        for col in number_cols:
            if col not in song_data or pd.isna(song_data[col]):
                song_data[col] = 0  
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])
    
def get_mean_vector(song_list, spotify_data):
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        if len(song_vector) == len(number_cols):  # Ensure the vector is complete
            song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = StandardScaler().fit(spotify_data[number_cols])
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


def song_form(request):
    context = {'range': range(1, 11)}
    return render(request, 'forms.html', context)


def submit_songs(request):
    if request.method == 'POST':
        songs = []
        for i in range(1, 11):
            name = request.POST.get(f'song{i}')
            year = request.POST.get(f'year{i}')
            try:
                year = int(year)
            except ValueError:
                return HttpResponse(f"Invalid year for song {i}", status=400)
            songs.append({'name': name, 'year': year})
        # Load CSV files
        spotify_data = load_csv('data.csv')
        data_by_artist = load_csv('data_by_artist.csv')
        data_by_genres = load_csv('data_by_genres.csv')
        data_by_year = load_csv('data_by_year.csv')
        data_w_genres = load_csv('data_w_genres.csv')
        
        # Recommend songs
        recommendations = recommend_songs(songs, spotify_data)
        
        recommendations_html = pd.DataFrame(recommendations).to_html(index=False)
        
        return HttpResponse(f"<h1>Recommendations</h1>{recommendations_html}")
    return HttpResponse("Invalid request method.")
