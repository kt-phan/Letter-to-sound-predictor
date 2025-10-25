import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def singular_form(word):
    word = word.lower().strip()
    if word.endswith("'s"):
        return word[:-2]
    return word

def clean_pronunciation(pron):
    # Remove any comment like "# foreign french" (and anything after '#')
    pron = re.sub(r'#.*', '', pron)

    # Remove digits from phonemes (e.g. AH0 → AH)
    pron = re.sub(r'([A-Z]+)\d', r'\1', pron)

    # Clean up extra whitespace
    pron = pron.strip()
    return pron


input_file = "cmu_dict_cleaned.csv"
output_file = "cmu_dict_cleaned_filtered.csv"

seen = set()

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')

        word = parts[0]

        # skip any entries ending with "'s"
        if word.endswith("'s"):
            continue

        word = singular_form(word)


        base = lemmatizer.lemmatize(word, pos='n')
        base = singular_form(base)


        if base in seen:
            continue

        seen.add(word)

        # removes numbers from phonemes
        pronunciation = clean_pronunciation(parts[1])

        if len(word) > 3 and word.isalpha():
            parts[1] = pronunciation
            outfile.write(','.join(parts) + "\n")

print("Filtering complete. Output written to", output_file)
