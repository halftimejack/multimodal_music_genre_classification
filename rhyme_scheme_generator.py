import pandas as pd
import re
import nltk
import string
import sys

# Change for other file sources/destinations
INPUT_FILE = 'balanced_genre_large.csv' 
OUTPUT_FILE = 'genre_data_large_with_schemes.csv'

try:
    nltk.data.find('corpora/cmudict.zip')
    phonetic_dict = nltk.corpus.cmudict.dict()
except LookupError:
    print("Downloading CMUDict...")
    nltk.download('cmudict')
    phonetic_dict = nltk.corpus.cmudict.dict()

def get_rhyming_part(word):
    if word not in phonetic_dict:
        return None
    pronunciation = phonetic_dict[word][0]
    for i, phone in enumerate(pronunciation):
        if phone[-1] in ('1', '2'):
            return tuple(pronunciation[i:])
    return tuple(pronunciation)

def get_scheme_label(index):
    alphabet = string.ascii_uppercase
    if index < 26:
        return alphabet[index]
    else:
        first_idx = (index // 26) - 1
        second_idx = index % 26
        first = alphabet[first_idx % 26]
        second = alphabet[second_idx]
        return first + second

def calculate_rhyme_scheme(row):
    lyrics = row['lyrics']
    
    if not isinstance(lyrics, str): return None
    
    lines = lyrics.strip().split('\n')
    end_words = []
    for line in lines:
        words = re.findall(r'\b\w+\b', line.lower())
        if words:
            end_words.append(words[-1])
            
    if not end_words:
        return None

    sound_to_label = {}
    next_label_idx = 0
    scheme_sequence = []
    
    for word in end_words:
        sound = get_rhyming_part(word)
        if sound is None:
            sound = ("OOV", word) 
            
        if sound in sound_to_label:
            scheme_sequence.append(sound_to_label[sound])
        else:
            if next_label_idx >= 702: 
                print(f"Skipping outlier song: '{row.get('title', 'Unknown')}' by {row.get('artist', 'Unknown')}")
                return None

            new_label = get_scheme_label(next_label_idx)
            sound_to_label[sound] = new_label
            scheme_sequence.append(new_label)
            next_label_idx += 1
            
    return " ".join(scheme_sequence)

print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    if 'genre' in df.columns:
        genre_col = 'genre'
    else: 
        genre_col = 'tag'
    
    if 'artist' not in df.columns: df['artist'] = 'Unknown'
    if 'title' not in df.columns: df['title'] = 'Unknown'

    cols_to_check = ['lyrics', genre_col]
    if 'chords' in df.columns:
        cols_to_check.append('chords')
        
    df = df.dropna(subset=cols_to_check)
    
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
    exit()

print("Generating Rhyme Schemes...")
df['rhyme_scheme'] = df.apply(calculate_rhyme_scheme, axis=1)

initial_len = len(df)
df = df.dropna(subset=['rhyme_scheme'])
df = df[df['rhyme_scheme'].str.len() > 0]
dropped_count = initial_len - len(df)

print(f"\n--- Processing Complete ---")
print(f"Dropped {dropped_count} songs.")
print(f"Remaining songs: {len(df)}")
print("\n--- Example Schemes ---")
print(df[[genre_col, 'rhyme_scheme']].head(10))

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved clean dataset with schemes to '{OUTPUT_FILE}'")