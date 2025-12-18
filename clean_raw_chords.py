import pandas as pd
import re

INPUT_FILE = 'chord_data_raw.csv'
OUTPUT_FILE = 'chord_data_for_training.csv'

print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    
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
    df['simple_genre'] = df.apply(map_genre, axis=1)
    
    df_clean = df.dropna(subset=['simple_genre'])
    
    df_clean[['chords', 'simple_genre']].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved strict training data to '{OUTPUT_FILE}'.")
    print(f"Song count: {len(df_clean)}")
    print(df_clean['simple_genre'].value_counts())

except Exception as e:
    print(f"Error: {e}")