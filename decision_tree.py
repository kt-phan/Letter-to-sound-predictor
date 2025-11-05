import pandas as pd

path = 'cmu_dict_cleaned_filtered_aligned.csv'
data = pd.read_csv(path)

def parse_alignment(alignment):
    try:
        pairs = eval(alignment)
        return [(g, p) for g, p in pairs if p != '#'] 
    except:
        return []
    

rows = []
context_size = 2
for _, row in data.iterrows():
    word = row['Word']
    alignment = parse_alignment(row['Alignment'])
    letters = [g for g, _ in alignment]
    phonemes = [p for _, p in alignment]

    for i in range(len(letters)):
        # context window
        left = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
        right = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
        feature = left[::-1] + [letters[i]] + right
        rows.append({"features": feature, "target": phonemes[i]})

expanded_df = pd.DataFrame(rows)
expanded_df.to_json("expanded_dataset.json", orient="records", lines=True)

