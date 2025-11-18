"""
Purpose: Utility functions for Grapheme-to-Phoneme feature engineering.

Includes:
- Alignment parsing
- Context window feature extraction (with Word ID)
- Feature encoding functions
- Target encoding functions
"""

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

# --- 1. Core Extraction ---

def parse_alignment(alignment_string):
    """
    Safely parse an alignment string into a list of (grapheme, phoneme) tuples.
    """
    try:
        return ast.literal_eval(alignment_string)
    except (ValueError, SyntaxError):
        return []

def extract_features(data_df, context_size=2):
    """
    Transforms the raw CMU DataFrame into a feature-target DataFrame.
    
    This is the main function for this module.
    
    Args:
        data_df (pd.DataFrame): The DataFrame loaded from the raw CSV.
        context_size (int): The window size (e.g., 2 for +/- 2 context).

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      ['features', 'target', 'word_id', 'original_word']
                      - 'features': a list of raw grapheme strings (e.g., ['_', 'a', 'b', 'e', 'r'])
                      - 'target': the raw phoneme string (e.g., 'B')
                      - 'word_id': the unique index of the word
                      - 'original_word': the word string
    """
    print('\nExtracting features from raw data...')
    feature_target_pairs = []
    boundary_marker = '_'
    
    for word_id, row in data_df.iterrows():
        word = row['Word']
        alignment = parse_alignment(row['Alignment'])
        if not alignment:
            continue
            
        letters = [g for g, _ in alignment]
        phonemes = [p for _, p in alignment]

        for i in range(len(letters)):
            left_context = [letters[i - j - 1] if i - j - 1 >= 0 else boundary_marker for j in range(context_size)]
            right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else boundary_marker for j in range(context_size)]
            
            feature = left_context[::-1] + [letters[i]] + right_context
            
            feature_target_pairs.append({
                "features": feature,
                "target": phonemes[i],
                "word_id": word_id,
                "original_word": word
            })

    print(f"Feature extraction done! Generated {len(feature_target_pairs)} examples.")
    return pd.DataFrame(feature_target_pairs)


# --- 2. Feature Encoding ---

def get_grapheme_encoder():
    """
    Creates and fits a LabelEncoder on a fixed grapheme alphabet.
    This ensures 'a' is always encoded to the same integer.
    
    Returns:
        sklearn.preprocessing.LabelEncoder: A *fitted* LabelEncoder for graphemes.
    """
    letters = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
               'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
               'w', 'x', 'y', 'z']
    
    grapheme_encoder = LabelEncoder()
    grapheme_encoder.fit(letters)
    print("Created fitted grapheme LabelEncoder.")
    return grapheme_encoder

def encode_features(feature_series, grapheme_encoder):
    """
    Encodes the 'features' column (a Series of lists) using a fitted LabelEncoder.
    
    Returns:
        np.ndarray: A 2D NumPy array of the label-encoded features.
    """
    print("Applying LabelEncoder to features...")
    # Use .transform() on each list in the Series
    X_encoded_list = [grapheme_encoder.transform(f) for f in feature_series]
    
    # Convert the list of arrays into a single 2D NumPy array
    return np.stack(X_encoded_list)


# --- 3. Target Encoding  ---

def fit_phoneme_encoder(y_train_series):
    """
    Fits a LabelEncoder on the training target data (phonemes).
    
    Returns:
        LabelEncoder: A *fitted* LabelEncoder object for the target.
    """
    print("Fitting phoneme encoder on training data...")
    phoneme_encoder = LabelEncoder()
    phoneme_encoder.fit(y_train_series)
    return phoneme_encoder

def encode_target(y_series, phoneme_encoder):
    """
    Encodes a phoneme target Series (y_train or y_test) using the fitted encoder.
    """
    return phoneme_encoder.transform(y_series)

def decode_target(y_encoded, phoneme_encoder):
    """
    Decodes encoded phoneme labels back into their string representation.
    """
    return phoneme_encoder.inverse_transform(y_encoded)