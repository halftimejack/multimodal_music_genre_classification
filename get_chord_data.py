import pandas as pd
from datasets import load_dataset

print("Downloading Chordonomicon from Hugging Face...")
try:
    dataset = load_dataset("ailsntua/Chordonomicon", split="train")
    df = dataset.to_pandas()
    
    print(f"Original size: {len(df)} songs")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'chords' in df.columns:
        df = df.dropna(subset=['chords'])
        print(f"Size after removing null chords: {len(df)}")
    
    genre_cols_to_check = []
    if 'main_genre' in df.columns:
        genre_cols_to_check.append('main_genre')
    if 'genres' in df.columns:
        genre_cols_to_check.append('genres')    
    if genre_cols_to_check:
        df = df.dropna(subset=genre_cols_to_check, how='all')
        print(f"Size after removing null genres: {len(df)}")

    print("\n--- 'main_genre' Value Counts ---")
    if 'main_genre' in df.columns:
        print(df['main_genre'].value_counts())
    
    print("\n--- 'genres' Value Counts (Top 20) ---")
    if 'genres' in df.columns:
        print(df['genres'].value_counts().head(20))
    
    cols_to_save = ['spotify_song_id', 'main_genre', 'genres', 'chords', 'spotify_artist_id']
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    
    df[cols_to_save].to_csv('chord_data_raw.csv', index=False)
    print(f"\nSaved raw data to 'chord_data_raw.csv' with {len(df)} rows.")

except Exception as e:
    print(f"Error: {e}")