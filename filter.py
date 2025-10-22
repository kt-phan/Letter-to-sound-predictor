import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def singular_form(word):
    word = word.lower().strip()
    if word.endswith("'s"):
        return word[:-2]
    return word


input_file = "cmu_dict_cleaned.csv"
output_file = "cmu_dict_cleaned_filtered.csv"

seen = set()

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        word = line.split(',')[0]
        word = singular_form(word)


        base = lemmatizer.lemmatize(word, pos='n')
        base = singular_form(base)


        if base in seen:
            continue

        seen.add(word)

        if len(word) > 3 and word.isalpha():
            outfile.write(line + "\n")

print("Filtering complete. Output written to", output_file)
