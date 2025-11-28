# Grapheme-to-Phoneme (G2P) Converter  
**Group 14 – CMPT 310 D200**  
**Members:**  
- Jovin Dosanjh (psd18@sfu.ca)  
- Arash Edalatpanah (aea53@sfu.ca)  
- David Lowe (drlowe@sfu.ca)  
- Toan Phan (tkp10@sfu.ca)  

---

## Project Overview
Our project aims to build a **grapheme-to-phoneme (G2P) converter**, which predicts how a written word should be pronounced.
The system takes a **word as input** and outputs a **sequence of phonemes** by classifying each letter based on its surrounding context.
We implement and compare four models: **Decision Tree**, **K-Nearest Neighbors (KNN)**, **Hybrid Rule-Based**, and **Hidden Markov Model (HMM)**, all trained on data from the **CMU Pronouncing Dictionary**.

---

## Environment Setup

In your project folder, run the following to create the virtual environment (only needs to happen once on set-up):
```bash
python -m venv .venv
```
To activate the virtual environment on subsequent logins you must run the below commands:

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

Finally, to install any necessary libraries:
```bash
pip install -r requirements.txt
```
To close the virtual environment, run:
```bash
deactivate
```
---
## Pipeline
Follow these steps in order to reproduce our results.

## Download Data
We use the CMU Pronouncing Dictionary hosted on Kaggle. This script authenticates using the Kaggle API and downloads the raw dataset to the data/ folder. Note: You must have your kaggle.json credentials set up for this to work.
```bash
python pipeline/00_download_cmu_dict.py
```
## Initial Processing
This script reads the raw dictionary file (cmudict.dict), parses the proprietary format, and converts it into a structured CSV. It handles basic cleaning, such as removing alternate pronunciations (e.g., keeping only the first variant of a word) and lowercasing the text.

Input: data/cmu_dict/cmudict.dict Output: data/processed_data/cmu_dict_cleaned.csv

```bash
python pipeline/01a_process_cmu_dict.py
```

## Filtering
This script processes the cleaned data to create a non-redundant dictionary. It normalizes words to their singular noun base form (e.g., keeping "CAT" and skipping "CATS"), removes numerical stress markers from phonemes (e.g., AH0 -> AH), and filters out possessives ('s) and words shorter than four letters to reduce noise.

```bash
python pipeline/01b_filter.py
```

## Alignment
We implement automatic one-to-one alignment based on descriptions provided by Damper et al. (2005). Our aligner uses dynamic programming (Needleman-Wunsch) refined by an Expectation-Maximization (EM) loop.

First, run the EM step to iteratively compute the association matrix until convergence. This matrix defines the probability of any letter mapping to any sound.

```bash
python pipeline/02a_em_step.py
```
Next, use the generated association matrix to perform the final alignment on the entire dataset. This creates the grapheme-phoneme pairs required for training.

```bash
python pipeline/02b_align.py
```

## Training, Evaluation, and Analysis
This master script handles the remainder of the pipeline. It automatically:

1. Converts aligned words into sliding context windows (size ±2) and integer-encodes them.

2. Trains four separate models:
#### 1. Decision Tree (Baseline)
#### 2. K-Nearest Neighbors (KNN)
#### 3. Hybrid (Rule-Based + DT)
#### 4. Hidden Markov Model (HMM)

Evaluation: Tests all models on a consistent 1,000-word subset.

Analysis: Generates performance plots (accuracy bars, word length analysis) and error reports in data/analysis_output/.

```bash
python pipeline/03_train_eval_analyze.py
```


---
## Citations
Black, A. W., Lenzo, K., & Pagel, V. (1998). Issues in building general letter to sound rules.

Damper, R. I., Marchand, Y., Marsters, J. D., & Bazin, A. I. (2005). Aligning text and phonemes for speech technology applications using an EM-like algorithm. International Journal of Speech Technology, 8(2), 147-160.


