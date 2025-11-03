'''
Purpose: Implements the Expectation-Maximization (EM) loop for
         Grapheme-Phoneme alignment, utilizing DP utilities.
Output: An association matrix A used to perform G2P alignment
'''
import pandas as pd
import numpy as np
from symbols import CMU_SYMBOLS, GRAPHEME_LIST
from dp_utilities import needleman_wunsch, trace_back
MATRIX_FILE = 'association_matrix.npy'

NUM_PHONEMES = len(CMU_SYMBOLS)
NUM_GRAPHEMES = len(GRAPHEME_LIST)
NULL_GRAPHEME = '$'
NULL_PHONEME = '#'
GRAPHEME_TO_IDX = {g: i for i, g in enumerate(GRAPHEME_LIST)}
PHONEME_TO_IDX = {p: i for i, p in enumerate(CMU_SYMBOLS)}

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

def initialize_table(data):    
    '''naive initialization of general association matrix A:
    - Every time grapheme 'g' and phoneme 'p' appear in 
    the same word, then A[g,p] is incremented
    '''
    A = np.zeros((NUM_GRAPHEMES, NUM_PHONEMES))
    for i, row in data.iterrows():
        word = row['Word']
        my_graphemes = list(word)      
        phoneme_str = row['Pronunciation']
        my_phonemes = phoneme_str.split()
                
        for g in my_graphemes:
            g_idx = GRAPHEME_TO_IDX[g]
            for p in my_phonemes:
                p_idx = PHONEME_TO_IDX[p]
                A[g_idx, p_idx] += 1
    return A 

def update_table(data, old_matrix):
    '''
    Use the old association matrix to align all words/phoneme strings,
    then use that alignment to make a new
    association matrix
    '''
    new_matrix = np.zeros((NUM_GRAPHEMES, NUM_PHONEMES))
    
    for _, row in data.iterrows():
        word = row['Word']
        phonemes = row['Pronunciation']
        path_matrix = needleman_wunsch(word, phonemes, old_matrix)
        # list of tuples with the form (grapheme, phoneme)
        alignment = trace_back(word, phonemes, path_matrix)

        # use alignment to update matrix
        for (grapheme, phoneme) in alignment:
            if grapheme in GRAPHEME_LIST and phoneme in CMU_SYMBOLS:
                g_idx = GRAPHEME_TO_IDX[grapheme]
                p_idx = PHONEME_TO_IDX[phoneme]
                new_matrix[g_idx, p_idx] += 1
    
    return new_matrix

def has_converged(old_matrix, new_matrix, tolerance):
    '''helper fucntion for em_allignment
    True if two arrays are close enough'''
    return np.allclose(old_matrix, new_matrix, rtol=1e-05)
        
def em_step(data):
    """
    - Main function to run the iterative EM-like alignment process.
    - Returns A, which is a matrix with entries indicating the degree 
    of association between grapheme g and phoneme p
    """
    A0 = initialize_table(data)
    A_new = A0.copy()
    
    # iteratively compute A until covergence
    i = 0 # keeps track of iterations
    print("Performing EM on association matrix A...")
    while True:
        A_old = A_new.copy()
        A_new = update_table(data, A_old)  
        print(f'--- Iteration {i} ---')
        print(f'Total score of A = {A_old.sum()}') 
        # print(f'A new sum = {A_new.sum()}') 
        i += 1
        if has_converged(A_old, A_new, tolerance=1e-6):
            print('A has converged!')
            break   
    
    return A_new

def main():
    # load data
    path = '../cmu_dict_cleaned_filtered.csv'
    data = load_data(path)  
    
    # perform Expectation-Maximization
    association_matrix = em_step(data)
    np.save(MATRIX_FILE, association_matrix)
    print(f"Successfully saved the final association matrix to {MATRIX_FILE}.")
    
if __name__ == "__main__":
    main()