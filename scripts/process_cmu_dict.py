# process_cmu_dict.py

import os
import pandas as pd

# 1️⃣ Set paths
input_file = "./cmu_dict/cmudict.dict"      # Path to your downloaded CMU dictionary
output_file = "./cmu_dict_cleaned.csv"      # Where the cleaned CSV will be saved

# 2️⃣ Read the dictionary
words = []
pronunciations = []

with open(input_file, 'r', encoding='latin-1') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(";;;"):  # skip comments and empty lines
            continue
        parts = line.split()  # split by whitespace
        word = parts[0]
        pron = ' '.join(parts[1:])
        words.append(word)
        pronunciations.append(pron)

# 3️⃣ Create a DataFrame
cmu_df = pd.DataFrame({"Word": words, "Pronunciation": pronunciations})

# 4️⃣ Clean the DataFrame

# a) Remove alternate pronunciations (WORD(1), WORD(2), etc.)
cmu_df['BaseWord'] = cmu_df['Word'].str.replace(r"\(\d+\)", "", regex=True)

# b) Keep only the first pronunciation for each word
cmu_df = cmu_df.drop_duplicates(subset='BaseWord', keep='first')

# c) Convert words to lowercase
cmu_df['BaseWord'] = cmu_df['BaseWord'].str.lower()

# d) Split pronunciation into list of phonemes
cmu_df['Phonemes'] = cmu_df['Pronunciation'].str.split()

# 5️⃣ Save cleaned CSV
cmu_df.to_csv(output_file, index=False)
print(f"Cleaned CMU dictionary saved to: {output_file}")

# 6️⃣ Preview
print("First 10 rows:")
print(cmu_df.head(10))
