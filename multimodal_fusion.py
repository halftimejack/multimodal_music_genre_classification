import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizer, DistilBertModel,
    RobertaConfig, RobertaModel,
    Trainer, TrainingArguments, PreTrainedTokenizerFast
)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import Dataset
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Ran in Google Colab

USE_TEXT = True
USE_CHORD = True
USE_RHYME = True

# Change file and directory for new runs
DATA_FILE = 'multimodal_with_schemes.csv'
OUTPUT_DIR = "./fusion_all_three_results"

# Change hyperparameters for new runs
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 5

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER = "/content/drive/MyDrive/NLP_Project/fusion_all_three_results"

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        OUTPUT_DIR = DRIVE_FOLDER
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

        drive_data_path = '/content/drive/MyDrive/NLP_Project/' + DATA_FILE
        if not os.path.exists(DATA_FILE) and os.path.exists(drive_data_path):
            print(f"Copying data from Drive...")
            shutil.copy(drive_data_path, DATA_FILE)
    except ImportError:
        print("Not running in Colab or Drive import failed.")

print(f"Loading {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    genre_col = 'genre' if 'genre' in df.columns else 'simple_genre' if 'simple_genre' in df.columns else 'tag'
    required_cols = [genre_col]
    if USE_TEXT: required_cols.append('lyrics')
    if USE_CHORD: required_cols.append('chords')
    if USE_RHYME: required_cols.append('rhyme_scheme')
    
    original_len = len(df)
    df = df.dropna(subset=required_cols)
    print(f"Data loaded. Rows retained: {len(df)}/{original_len}")
except FileNotFoundError:
    print(f"Error: File not found.")
    exit()

labels = sorted(df[genre_col].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
df['label'] = df[genre_col].map(label2id)

print("Loading Text Tokenizer...")
text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

chord_tokenizer = None
if USE_CHORD:
    print("Training Chord Tokenizer...")
    chord_tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    chord_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"], vocab_size=5000)
    chord_tok.train_from_iterator(df['chords'].tolist(), trainer=trainer)
    chord_tok.save("all_chord.json")
    chord_tokenizer = PreTrainedTokenizerFast(tokenizer_file="all_chord.json", pad_token="<pad>")

rhyme_tokenizer = None
if USE_RHYME:
    print("Training Rhyme Tokenizer...")
    rhyme_tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    rhyme_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"], vocab_size=1000)
    rhyme_tok.train_from_iterator(df['rhyme_scheme'].tolist(), trainer=trainer)
    rhyme_tok.save("all_rhyme.json")
    rhyme_tokenizer = PreTrainedTokenizerFast(tokenizer_file="all_rhyme.json", pad_token="<pad>")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_fn(examples):
    output = {'labels': examples['label']}
    
    if USE_TEXT:
        text_enc = text_tokenizer(examples['lyrics'], padding="max_length", truncation=True, max_length=128)
        output['input_ids'] = text_enc['input_ids']
        output['attention_mask'] = text_enc['attention_mask']
        
    if USE_CHORD:
        chord_enc = chord_tokenizer(examples['chords'], padding="max_length", truncation=True, max_length=128)
        output['chord_ids'] = chord_enc['input_ids']
        output['chord_mask'] = chord_enc['attention_mask']
        
    if USE_RHYME:
        rhyme_enc = rhyme_tokenizer(examples['rhyme_scheme'], padding="max_length", truncation=True, max_length=128)
        output['rhyme_ids'] = rhyme_enc['input_ids']
        output['rhyme_mask'] = rhyme_enc['attention_mask']

    return output

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

cols = ['labels']
if USE_TEXT: cols.extend(['input_ids', 'attention_mask'])
if USE_CHORD: cols.extend(['chord_ids', 'chord_mask'])
if USE_RHYME: cols.extend(['rhyme_ids', 'rhyme_mask'])

train_dataset.set_format(type='torch', columns=cols)
test_dataset.set_format(type='torch', columns=cols)

class TripleFusion(nn.Module):
    def __init__(self, num_labels, use_text=True, use_chord=True, use_rhyme=True):
        super().__init__()
        self.use_text = use_text
        self.use_chord = use_chord
        self.use_rhyme = use_rhyme

        if self.use_text:
            self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if self.use_chord:
            c_config = RobertaConfig(vocab_size=5000, hidden_size=512, num_hidden_layers=6, num_attention_heads=8)
            self.chord_model = RobertaModel(c_config)

        if self.use_rhyme:
            r_config = RobertaConfig(vocab_size=1000, hidden_size=512, num_hidden_layers=6, num_attention_heads=8)
            self.rhyme_model = RobertaModel(r_config)

        fusion_dim = 0
        if self.use_text: fusion_dim += 768
        if self.use_chord: fusion_dim += 512
        if self.use_rhyme: fusion_dim += 512

        if fusion_dim == 0:
            raise ValueError("At least one modality must be set to True")

        self.classifier = nn.Linear(fusion_dim, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, labels=None, input_ids=None, attention_mask=None, chord_ids=None, chord_mask=None, rhyme_ids=None, rhyme_mask=None):
        vectors_to_fuse = []
        
        if self.use_text:
            text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            vectors_to_fuse.append(text_out.last_hidden_state[:, 0, :])
            
        if self.use_chord:
            chord_out = self.chord_model(input_ids=chord_ids, attention_mask=chord_mask)
            vectors_to_fuse.append(chord_out.last_hidden_state[:, 0, :])
            
        if self.use_rhyme:
            rhyme_out = self.rhyme_model(input_ids=rhyme_ids, attention_mask=rhyme_mask)
            vectors_to_fuse.append(rhyme_out.last_hidden_state[:, 0, :])

        combined = torch.cat(vectors_to_fuse, dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

model = TripleFusion(num_labels=len(labels), use_text=USE_TEXT, use_chord=USE_CHORD, use_rhyme=USE_RHYME)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
    compute_metrics=lambda p: {"acc": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1)),
                               "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1), average='macro')}
)

print("Starting Training...")
trainer.train()

metrics = trainer.evaluate()
print(f"\nFINAL RESULTS: Acc: {metrics['eval_acc']:.4f}, F1: {metrics['eval_f1']:.4f}")

preds = np.argmax(trainer.predict(test_dataset).predictions, axis=-1)
report = classification_report(test_dataset['labels'], preds, target_names=labels, digits=4)
print(report)

# Change file names for new runs
if USE_GOOGLE_DRIVE:
    with open(os.path.join(OUTPUT_DIR, 'report_all_three.txt'), 'w') as f: f.write(report)
    cm = confusion_matrix(test_dataset['labels'], preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels)
    plt.title('Fusion Matrix (All Three)')
    plt.savefig(os.path.join(OUTPUT_DIR, "matrix_all_three.png"))