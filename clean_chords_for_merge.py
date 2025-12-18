import pandas as pd
import re

# This file has the same logic as clean_raw_chords but it specifically keeps all potential merges instead of aggressively dropping songs

INPUT_FILE = 'chords_with_metadata.csv' 
OUTPUT_FILE = 'chords_enriched_cleaned.csv'

print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    
    print("Cleaning chord strings...")
    def clean_chord_string(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'<.*?>', '', text) # Remove <verse> tags
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['chords'] = df['chords'].apply(clean_chord_string)
    df = df[df['chords'].str.len() > 0]
    
    # These genre mappings are superceded by the lyric genre mappings after merging
    # They are only used for training on the unmerged data in the scale experiment
    def map_genre(row):
        g = str(row['main_genre']).lower()
        if g == 'pop': return 'pop'
        if g == 'rock': return 'rock'
        if g == 'country': return 'country'
        if g == 'rap': return 'rap'
        if g == 'soul': return 'rb'
        if g == 'metal': return 'rock'
        if g == 'punk': return 'rock'
        if g == 'alternative': return 'rock'
        if g == 'pop rock': return 'pop'
        return None 

    print("Mapping genres...")
    df['chord_source_genre'] = df.apply(map_genre, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved  merge data to '{OUTPUT_FILE}'.")
    print(f"Song count: {len(df)}")

except Exception as e:
    print(f"Error: {e}")