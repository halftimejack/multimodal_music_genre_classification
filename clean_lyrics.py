import pandas as pd
import re
import os

INPUT_FILE = 'song_lyrics.csv'
OUTPUT_FILE = 'lyrics_cleaned.csv'
CHUNK_SIZE = 50000

def clean_lyric_text(text):
    if not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    return cleaned_text.strip()

def process_and_clean_dataset():
    print(f"Starting full dataset cleaning...")
    print(f"Input: '{INPUT_FILE}'")
    print(f"Output: '{OUTPUT_FILE}'")

    try:
        chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"--- ERROR: Input file '{INPUT_FILE}' not found. ---")
        return

    total_rows_processed = 0
    total_rows_written = 0
    is_first_chunk = True

    for i, chunk in enumerate(chunk_iterator):
        total_rows_processed += len(chunk)
        
        if 'language' not in chunk.columns or 'tag' not in chunk.columns:
            print(f"\n--- ERROR ---")
            print(f"The 'language' or 'tag' column was not found in '{INPUT_FILE}'.")
            return

        chunk['lyrics'] = chunk['lyrics'].apply(clean_lyric_text)
        cleaned_chunk = chunk[chunk['language'] == 'en']
        cleaned_chunk = cleaned_chunk.dropna(subset=['tag'])
        cleaned_chunk = cleaned_chunk[cleaned_chunk['tag'] != 'misc']
        final_chunk = cleaned_chunk[cleaned_chunk['lyrics'].str.strip().astype(bool)]
        
        if not final_chunk.empty:
            current_rows_written = len(final_chunk)
            total_rows_written += current_rows_written
            
            if is_first_chunk:
                final_chunk.to_csv(OUTPUT_FILE, index=False, mode='w', header=True)
                is_first_chunk = False
            else:
                final_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
        
        print(f"  - Processed chunk {i + 1}: Kept {len(final_chunk)} of {len(chunk)} rows.")

    print("\n--- Cleaning Complete ---")
    print(f"Total songs processed: {total_rows_processed:,}")
    print(f"Total songs written to new file: {total_rows_written:,}")
    
    try:
        original_size = os.path.getsize(INPUT_FILE) / (1024**2)
        cleaned_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
        print(f"Original file size ('{INPUT_FILE}'): {original_size:.2f} MB")
        print(f"Final cleaned file size ('{OUTPUT_FILE}'): {cleaned_size:.2f} MB")
    except (FileNotFoundError, ZeroDivisionError):
        pass
    
    print(f"\n'{OUTPUT_FILE}' is now ready for all future analysis.")

if __name__ == "__main__":
    process_and_clean_dataset()