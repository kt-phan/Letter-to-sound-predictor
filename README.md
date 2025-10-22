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

In your project folder, run:

```bash
python -m venv .venv

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt


