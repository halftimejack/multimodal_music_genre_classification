import pandas as pd

INPUT_FILE = 'multimodal_dataset.csv'
OUTPUT_FILE = 'balanced_multimodal_train.csv'
SAMPLES_PER_GENRE = 4000

print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    
    genre_col = 'genre'

    print(f"Balancing to {SAMPLES_PER_GENRE} per genre...")
    
    balanced_df = df.groupby(genre_col).apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_GENRE), random_state=42)
    ).reset_index(drop=True)
    
    if 'chords' not in balanced_df.columns:
        print("WARNING: 'chords' column missing!")
    if 'lyrics' not in balanced_df.columns:
        print("WARNING: 'lyrics' column missing!")

    balanced_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Saved '{OUTPUT_FILE}' with {len(balanced_df)} songs.")
    print(balanced_df[genre_col].value_counts())

except Exception as e:
    print(f"Error: {e}")