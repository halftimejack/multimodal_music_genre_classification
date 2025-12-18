import pandas as pd

INPUT_FILE = 'lyrics_cleaned.csv'
OUTPUT_FILE = 'balanced_genre_small.csv'

SAMPLES_PER_GENRE = 86658
TARGET_GENRES = ['pop', 'rap', 'rock', 'rb', 'country']

print(f"Reading {INPUT_FILE}...")

try:
    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df = df[df['tag'].isin(TARGET_GENRES)]
    
    print(f"Total rows matching target genres: {len(df):,}")
    
    print(f"Sampling {SAMPLES_PER_GENRE:,} songs per genre...")
    balanced_df = df.groupby('tag').apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_GENRE), random_state=42)
    ).reset_index(drop=True)
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS! Created '{OUTPUT_FILE}'")
    print(f"Total Dataset Size: {len(balanced_df):,} songs")
    print("\nCounts per genre:")
    print(balanced_df['tag'].value_counts())

except Exception as e:
    print(f"Error: {e}")