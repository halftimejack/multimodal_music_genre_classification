import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os
import sys

INPUT_FILE = 'chord_data_raw.csv'
OUTPUT_FILE = 'chords_with_metadata.csv'
BATCH_SIZE = 50

CLIENT_ID = "ID_GOES_HERE"
CLIENT_SECRET = "SECRET_GOES_HERE"

FORCE_START_INDEX = None # Used for API timeouts

print("Authenticating with Spotify...")
try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception as e:
    print(f"Authentication Failed: {e}")
    exit()

def process_batch(ids):
    """Fetches metadata for a list of up to 50 Spotify IDs."""
    try:
        results = sp.tracks(ids)
        metadata = []
        for track in results['tracks']:
            if track:
                metadata.append({
                    'spotify_id': track['id'],
                    'spotify_artist': track['artists'][0]['name'],
                    'spotify_title': track['name']
                })
            else:
                metadata.append(None) 
        return metadata
    except Exception as e:
        raise e 

print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
    exit()

id_col = 'spotify_song_id'
if id_col not in df.columns:
    print(f"Error: Column '{id_col}' not found in input file.")
    exit()

start_idx = 0
if FORCE_START_INDEX is not None:
    start_idx = FORCE_START_INDEX
    print(f"Forcing start from input row {start_idx}...")
elif os.path.exists(OUTPUT_FILE):
    print(f"Found existing '{OUTPUT_FILE}'. Attempting to auto-resume...")
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            row_count = sum(1 for row in f) - 1 
        start_idx = row_count
        print(f"Auto-resume calculated row {start_idx}.")
    except Exception:
        print("Could not read output file. Starting from 0.")
else:
    headers = ['spotify_id', 'spotify_artist', 'spotify_title', 'chords', 'main_genre', 'genres']
    pd.DataFrame(columns=headers).to_csv(OUTPUT_FILE, index=False)

total_rows = len(df)
print(f"Starting processing. {total_rows - start_idx} rows remaining.")

current_idx = start_idx

try:
    for i in range(start_idx, total_rows, BATCH_SIZE):
        current_idx = i
        
        batch_df = df.iloc[i : i + BATCH_SIZE].copy()
        ids = batch_df[id_col].tolist()
        
        valid_ids = [x for x in ids if isinstance(x, str) and len(x) == 22]
        
        if valid_ids:
            try:
                meta_list = process_batch(valid_ids)
                
                if meta_list:
                    meta_dict = {m['spotify_id']: m for m in meta_list if m}
                    batch_result = []
                    for _, row in batch_df.iterrows():
                        sid = row[id_col]
                        if sid in meta_dict:
                            m = meta_dict[sid]
                            batch_result.append({
                                'spotify_id': sid,
                                'spotify_artist': m['spotify_artist'],
                                'spotify_title': m['spotify_title'],
                                'chords': row['chords'], 
                                'main_genre': row['main_genre'], 
                                'genres': row['genres']
                            })
                    
                    if batch_result:
                        pd.DataFrame(batch_result).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            except Exception as e:
                print(f"\nAPI Error at row {i}: {e}")
                raise e

        percent = ((i + BATCH_SIZE) / total_rows) * 100
        if i % 1000 == 0:
            print(f"Processed {i} / {total_rows} ({percent:.2f}%)")
        else:
            print(f"Processed {i} / {total_rows} ({percent:.2f}%)", end='\r') 
        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"\n\n!!! SCRIPT STOPPED BY USER AT ROW: {current_idx} !!!")

except Exception as e:
    print(f"\n\n!!! SCRIPT CRASHED AT ROW: {current_idx} !!!")
    print(f"Error: {e}")
    sys.exit()

print("\n\n--- PROCESSING COMPLETE ---")

print(f"Checking '{OUTPUT_FILE}' for duplicates...")
try:
    final_df = pd.read_csv(OUTPUT_FILE)
    initial_len = len(final_df)
    
    final_df = final_df.drop_duplicates(subset=['spotify_id'])
    final_len = len(final_df)
    
    if initial_len > final_len:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Cleaned up {initial_len - final_len} duplicate rows.")
    else:
        print("No duplicates found.")
        
    print(f"Final dataset size: {final_len} songs.")
    
except Exception as e:
    print(f"Error during cleanup: {e}")