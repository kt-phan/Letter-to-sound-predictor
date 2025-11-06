# Grapheme-to-Phoneme (G2P) Converter  
**Group 14 – CMPT 310 D200**  
**Members:**  
- Jovin Dosanjh (psd18@sfu.ca)  
- Arash Edalatpanah (aea53@sfu.ca)  
- David Lowe (drlowe@sfu.ca)  
- Toan Phan (tkp10@sfu.ca)  

---

## 🧠 Project Overview
Our project aims to build a **grapheme-to-phoneme (G2P) converter**, which predicts how a written word should be pronounced.  
The system takes a **word as input** and outputs a **sequence of phonemes** by classifying each letter based on its surrounding context.  
For this milestone, we implement a **Decision Tree classifier** trained on aligned letter–phoneme pairs extracted from the **CMU Pronouncing Dictionary**.

---

## ⚙️ Environment Setup
## 1. Create the Environment

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
## Filtering
## Alignment
## Machine Learning
### 1. Decision Trees
### 2. K-Nearest Neighbours

## CItations
Black, A. W., Lenzo, K., & Pagel, V. (1998). Issues in building general letter to sound rules.

Damper, R. I., Marchand, Y., Marsters, J. D., & Bazin, A. I. (2005). Aligning text and phonemes for speech technology applications using an EM-like algorithm. International Journal of Speech Technology, 8(2), 147-160.


