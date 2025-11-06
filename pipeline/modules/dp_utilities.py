'''
Purpose: Defines constants and core dynamic programming (DP)
         functions for Grapheme-Phoneme alignment.
'''
import pandas as pd
import numpy as np
from modules.symbols import CMU_SYMBOLS, GRAPHEME_LIST

NUM_PHONEMES = len(CMU_SYMBOLS)
NUM_GRAPHEMES = len(GRAPHEME_LIST)
NULL_GRAPHEME = '$'
NULL_PHONEME = '#'
GRAPHEME_TO_IDX = {g: i for i, g in enumerate(GRAPHEME_LIST)}
PHONEME_TO_IDX = {p: i for i, p in enumerate(CMU_SYMBOLS)}
NO_DIRECTION, DOWN, DIAGONAL, RIGHT = -1, 0, 1, 2 # for traceback of matrix  
DELTA = 0 # penalty for null grapheme/phoneme


def create_word_association_matrix(word, phonemes, general_association_matrix):
    '''
    Creates a word-specific association matrix (B) by copying relevant
    values from the general association matrix (A).
    A: The general (NUM_GRAPHEMES x NUM_PHONEMES) association matrix.
    '''
    A = general_association_matrix
    padded_graphemes = list(NULL_GRAPHEME + word + NULL_GRAPHEME)
    padded_phonemes = [NULL_PHONEME] + phonemes.split() + [NULL_PHONEME]
    row_len = len(padded_graphemes)
    col_len = len(padded_phonemes)
    # initialize word association matrix B
    B = np.zeros((row_len, col_len)) 
    for i in range(row_len):
        grapheme = padded_graphemes[i]  
        for j in range(col_len):
            phoneme = padded_phonemes[j]
            # copy A[g, p] into B[i, j] only if g and p are real symbols.   
            if grapheme != NULL_GRAPHEME and phoneme != NULL_PHONEME:
                    g_idx = GRAPHEME_TO_IDX[grapheme]
                    p_idx = PHONEME_TO_IDX[phoneme]
                    B[i, j] = A[g_idx, p_idx]
            # B[i, j] remains 0 if g = '$' or p = '#'

    return B

######### CORE ALGORITHM #############
def needleman_wunsch(word, phonemes, general_association_matrix):
    '''
    - Output: path matrix D for later traceback to find alignment
    - Word and phonemes are strings
    '''
    A = general_association_matrix 
    padded_graphemes = list(NULL_GRAPHEME + word + NULL_GRAPHEME)
    padded_phonemes = [NULL_PHONEME] + phonemes.split() + [NULL_PHONEME]
    row_len = len(padded_graphemes)
    col_len = len(padded_phonemes)
    
    B = create_word_association_matrix(word, phonemes, A)
    C = np.zeros((row_len, col_len)) # store acumulation of association values
    D = np.zeros((row_len, col_len)) # tracks path
    
    # Implicitly initialize first row/column (gap penalties of 0)
    # C[i, 0] = 0 for all i
    # C[0, j] = 0 for all j
    
    # inditialize directions 
    D[0,0] = NO_DIRECTION
    D[1:,0] = DOWN
    D[0,1:] = RIGHT
    
    # accumulation step
    for i in range(1, row_len):
        for j in range(1, col_len):
            # association value of current grapheme and phoneme
            association_value = B[i,j]
            
            # --- Options for the current cell C[i, j] ---
            # Option 0: Diagonal (Grapheme i aligns with Phoneme j)
            score_diagonal = C[i-1, j-1] + association_value
            # Option 1: Down (Grapheme i aligns with NULL Phoneme)
            score_down = C[i-1, j] - DELTA
            # Option 2: Right (NULL Grapheme aligns with Phoneme j)
            score_right = C[i, j-1] - DELTA
            max_score = max(score_diagonal, score_down, score_right)  
            C[i, j] = max_score    
            if max_score == score_diagonal:
                D[i][j] = DIAGONAL
            elif max_score == score_down:
                D[i][j] = DOWN
            else:
                D[i][j] = RIGHT
    return D

def trace_back(word, phonemes, path_matrix):
    """Traceback the D matrix to get the optimal alignment pairs.
    Returns a list of tuples: 
    [ (grapheme_1, phoneme_1), (grapheme_2, phoneme_2), ...]
    Assumes word, phonemes are strings
    """
    D = path_matrix
    row_len, col_len = D.shape
    last_row_idx = row_len
    last_col_idx = col_len
    i, j = last_row_idx - 1, last_col_idx - 1
    alignment_pairs = [] # list of tuples
    
    padded_graphemes = list(NULL_GRAPHEME + word + NULL_GRAPHEME)
    padded_phonemes = [NULL_PHONEME] + phonemes.split() + [NULL_PHONEME] 
    # Stop when reaching the origin C[0, 0]
    while D[i, j] != NO_DIRECTION:
        move = D[i, j]
        
        if move == DIAGONAL:  
            grapheme = padded_graphemes[i]
            phoneme = padded_phonemes[j]
            i -= 1
            j -= 1
        elif move == DOWN:  
            grapheme = padded_graphemes[i]
            phoneme = NULL_PHONEME
            i -= 1
        elif move == RIGHT:  # Right: (NULL_GRAPHEME, p_j)
            grapheme = NULL_GRAPHEME
            phoneme = padded_phonemes[j]
            j -= 1
            
        # Store the (grapheme, phoneme) pair
        alignment_pairs.append((grapheme, phoneme))
        
    alignment_pairs.reverse() # reverse to get the right order
    # remove last pair (NULL_GRAPHEME, NULL_PHONEME)
    return alignment_pairs[:-1] 


    


    


