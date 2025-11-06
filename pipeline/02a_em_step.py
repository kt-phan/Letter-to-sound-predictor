'''
Purpose: Implements the Expectation-Maximization (EM) loop for
         Grapheme-Phoneme alignment, utilizing DP utilities.
Output: An association matrix A used to perform G2P alignment
'''
import pandas as pd
import numpy as np
from modules.em_step import em_step, MATRIX_FILE
import sys 

def load_data(path):
    # load dataset that needs alignment
    # na = false to avoid capturing 'null' as float
    print("Loading data...")
    try:
        data = pd.read_csv(path, usecols=['Word', 'Pronunciation'], na_filter=False)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {path}")
        sys.exit(1)
    print('All words and phonemes successfully loaded. ')
    return data

def main():
    # load data
    path = 'data/processed_data/cmu_dict_cleaned_filtered.csv'
    data = load_data(path)  
    
    # perform Expectation-Maximization
    association_matrix = em_step(data)
    np.save(MATRIX_FILE, association_matrix)
    print(f"Successfully saved the final association matrix to {MATRIX_FILE}.")
    
if __name__ == "__main__":
    main()