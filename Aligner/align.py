'''
Purpose: Given an association matrix produced in em_step.py,
        aligns all word-phoneme pairs and store the results
'''
import pandas as pd
import numpy as np
from em_step import em_step, MATRIX_FILE
from dp_utilities import needleman_wunsch, trace_back

def load_data(path):
    # load dataset that needs alignment
    # na = false to avoid capturing 'null' as float
    print("Loading data...")
    try:
        data = pd.read_csv(path, usecols=['Word', 'Pronunciation'], na_filter=False)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {path}")
        return pd.DataFrame()
    print('All words and phonemes successfully loaded. ')
    return data

def align(row, association_matrix):
    '''
    Aligns a word and its corresponding phoneme string
    returns a list of tuples like so:
    [ (grapheme_1, phoneme_1), (grapheme_2, phoneme_2) ]
    '''
    A = association_matrix
    word = row['Word']
    phonemes = row['Pronunciation']
    path_matrix = needleman_wunsch(word, phonemes, A)
    # list of tuples with the form (grapheme, phoneme)
    alignment = trace_back(word, phonemes, path_matrix)
    return alignment

def store_data(data, output_path):
    print(f"Saving results to {output_path}...")
    data.to_csv(output_path, index=False)
    print("Alignment results successfully saved!")
    return
    
def main():
    # load data
    path = '../cmu_dict_cleaned_filtered.csv'
    data = load_data(path)  
    
    # load association matrix
    association_matrix = np.load(MATRIX_FILE)
    A = association_matrix
    
    # perform alignment
    print()
    print('Performing alignment...')
    data['Alignment'] = data.apply(align, axis=1, args=(A,))
    print('Alignment complete.')
    
    # takes out entries with the null grapheme (the new line)
    print('\nFiltering out alignments containing the null grapheme ($)...')
    # Filter rows where the 'Alignment' list contains a tuple with '$'
    data = data[
        data['Alignment'].apply(
            lambda alignment_list: not any('$' in pair for pair in alignment_list)
        )
    ].reset_index(drop=True)
    print('Filtering complete.')
    
    # store dataFrame
    print()
    output_path = '../cmu_dict_cleaned_filtered_aligned.csv'
    store_data(data, output_path)
    
if __name__ == "__main__":
    main()

