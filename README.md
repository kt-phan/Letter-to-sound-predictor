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
For this milestone, we implement a **Decision Tree classifier** trained on aligned letter–phoneme pairs extracted from the **CMU Pronouncing Dictionary**.

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

Finall, to install any necessary libraries:
```bash
pip install -r requirements.txt
```
To close the virtual environment, run:
```bash
deactivate
```
---
## Pipeline
### Download data
### Cleaning
### Filtering
This script processes the raw CMU Pronouncing Dictionary data (```cmu_dict_cleaned.csv```) to create a clean, non-redundant dictionary, outputting to  ```cmu_dict_cleaned_filtered.csv```. It first normalizes words  to their singular noun base form, ensuring only one entry per root word (e.g., keeping "CAT" and skipping "CATS"). It then cleans pronunciations by removing numerical stress markers (like AH0 to AH) and filters out possessives ('s) and words shorter than four letters.

Usage:
```
python pipeline/01b_filter.py
```
### Alignment
Here, we implemented automatic one-to-one alignment based on descriptions provided by Damper et al. (2005). Our aligner uses dynamic programming (specifically the Needleman-Wunsch algorithm) as well as  the Expectation-Maximum algorithm.
First, we can run the command below, which iteratively computes an association matrix A until convergence and save the natrix under the file ```association_matrix.np```. Each entry of A represents the degree of association between any grapheme-phoneme pair. 
```bash
python pipeline/02a_em_step.py
```
The matrix A is then used to generate alignments using dynamic programming, in which we try to maximize the total association scores. Running the following script will excecute the alignment process: 
```bash
python pipeline/02b_align.py
```
### Feature extraction
### Machine Learning
#### 1. Decision Trees
#### 2. K-Nearest Neighbours
---
## CItations
Black, A. W., Lenzo, K., & Pagel, V. (1998). Issues in building general letter to sound rules.

Damper, R. I., Marchand, Y., Marsters, J. D., & Bazin, A. I. (2005). Aligning text and phonemes for speech technology applications using an EM-like algorithm. International Journal of Speech Technology, 8(2), 147-160.


