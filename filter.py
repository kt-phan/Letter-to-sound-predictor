
input_file = "cmu_dict_cleaned.csv"
output_file = "cmu_dict_cleaned_filtered.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        first_word = line.split(',')[0]

        if len(first_word) < 6:
            outfile.write(line + "\n")

print("Filtering complete. Output written to", output_file)
