# Multimodal Music Genre Classification

**Author:** Jack Blake  
**Date:** December 2025  
**Project Type:** Data Engineering & Machine Learning Research

---

## 📌 Project Overview
This project investigates an automated framework for music genre classification using an **intermediate fusion architecture**. Unlike traditional unimodal models, this system integrates three distinct symbolic modalities:
* **Semantics:** Song lyrics.
* **Harmony:** Chord progressions.
* **Structure:** Rhyme schemes.

The study evaluates the distinct contribution of each modality and identifies synergies that allow for robust classification even in the absence of raw audio data.



---

## 🛠 Technical Architecture
The model uses **parallel transformer branches** to capture modality-specific features before concatenating them for final classification.

* **Lyrics Branch:** Fine-tuned **DistilBERT** base model.
* **Chords Branch:** Custom **RoBERTa encoder** trained from scratch (6 layers, 8 attention heads, hidden size 512) to optimize for harmonic sequences.
* **Rhyme Branch:** Custom RoBERTa architecture utilizing a 1,000-token vocabulary generated via the **CMU Pronouncing Dictionary**.
* **Fusion Layer:** Concatenates the [CLS] tokens from each branch into a single multimodal feature vector.

---

## 📊 Data Engineering Pipeline
A core component of this project was the construction of a large-scale multimodal corpus through complex data joining and cleaning.

### 1. Data Sources
* **Genius Lyrics:** 5.1 million entries (scraped textual data).
* **Chordonomicon:** 680,000 chord sequences parsed from crowdsourced charts.
* **Spotify API:** Used to retrieve standardized metadata (Artist/Title) for cross-dataset synchronization.

### 2. Processing Pipeline
* **Normalization:** Applied a strict pipeline (lowercasing, whitespace stripping, regex-based metadata removal) to ensure successful joining across disparate sources.
* **Inner Join:** Conducted a join on standardized metadata, resulting in a **200,000-song corpus** with full lyrical and harmonic data.
* **Stratification:** Balanced the final experimental dataset to **20,000 samples** to mitigate label skew toward Pop and Rock.



---

## 📈 Experiments and Results
An **ablation study** was conducted to isolate the predictive contribution of each symbolic modality.

| Model | Modalities | Accuracy | F1 Score |
| :--- | :---: | :---: | :---: |
| Baseline (Random) | 0 | 0.200 | 0.200 |
| Rhyme only | 1 | 0.364 | 0.332 |
| Chords only | 1 | 0.380 | 0.379 |
| Text only | 1 | 0.531 | 0.533 |
| **No-Text (Chords + Rhyme)** | **2** | **0.454** | **0.461** |
| Text + Rhyme | 2 | 0.547 | 0.546 |
| Text + Chords | 2 | 0.567 | 0.57 |
| **Full Fusion** | **3** | **0.574** | **0.575** |

### Key Insights
* **Symbolic Synergy:** Every added modality resulted in a strict increase in performance, validating that genre is a composite of semantics, harmony, and structure.
* **Architecture vs. Volume:** While massive unimodal datasets (430k samples) currently yield higher absolute scores, this **fusion architecture is highly data-efficient**, achieving competitive results (0.575 F1) with only 20k samples.
* **Genre Signatures:** The Rhyme-Only model peaked on **Rap (F1: 0.59)**, while the Chord-Only model successfully identified the distinct harmonic signatures of **Country and R&B**.

---

## 🚀 Future Work
* **Audio Integration:** Incorporating audio spectrograms to better distinguish between compositionally similar genres like Pop and Rock.
* **Linguistic Refinement:** Improving the rhyme modality to account for **slant rhymes** and near rhymes.

---

## 🔗 References
* [1] [Genius Song Lyrics Dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
* [2] [Chordonomicon Dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon)
* [3] [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
