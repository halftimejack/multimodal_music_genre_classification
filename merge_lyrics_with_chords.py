import pandas as pd
import re

LYRICS_FILE = 'lyrics_cleaned.csv'
CHORDS_FILE = 'chords_enriched_cleaned.csv'
OUTPUT_FILE = 'multimodal_dataset.csv'

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

print(f"Reading Lyrics from {LYRICS_FILE}...")
try:
    df_lyrics = pd.read_csv(LYRICS_FILE)
    df_lyrics = df_lyrics[['artist', 'title', 'lyrics', 'tag']]
    df_lyrics = df_lyrics.rename(columns={'tag': 'genre', 'artist': 'artist_lyric', 'title': 'title_lyric'})
    
    print(f"Reading Chords from {CHORDS_FILE}...")
    df_chords = pd.read_csv(CHORDS_FILE)
    df_chords = df_chords.rename(columns={
        'spotify_artist': 'artist_chord', 
        'spotify_title': 'title_chord',
    })
    
    print("Normalizing Lyric keys...")
    df_lyrics['key_artist'] = df_lyrics['artist_lyric'].apply(normalize_text)
    df_lyrics['key_title'] = df_lyrics['title_lyric'].apply(normalize_text)

    print("Normalizing Chord keys...")
    df_chords['key_artist'] = df_chords['artist_chord'].apply(normalize_text)
    df_chords['key_title'] = df_chords['title_chord'].apply(normalize_text)

    print("\n--- MERGING DATASETS ---")
    merged_df = pd.merge(
        df_lyrics, 
        df_chords, 
        on=['key_artist', 'key_title'], 
        how='inner'
    )

    merged_df = merged_df.drop_duplicates(subset=['key_artist', 'key_title'])
    
    target_genres = ['pop', 'rap', 'rock', 'rb', 'country']
    merged_df = merged_df[merged_df['genre'].isin(target_genres)]

    if len(merged_df) > 0:
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved multimodal dataset to '{OUTPUT_FILE}'")

except Exception as e:
    print(f"Error: {e}")