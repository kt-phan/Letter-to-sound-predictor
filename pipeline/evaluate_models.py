'''
Purpose: Centralized evaluation logic.
Calculates Per-Letter Accuracy and Per-Word Accuracy.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import word_evaluation as wle

def calculate_letter_accuracy(model, X_test, y_test):
    """
    Standard sklearn accuracy for isolated letter predictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred

def calculate_word_accuracy(test_df, model, grapheme_encoder, phoneme_encoder):
    """
    Uses word_evaluation.py to reconstruct full words and check correctness.
    """
    # wle.evaluate_model_word_level prints output, but also returns accuracy
    # We suppress the print output here to keep the main log clean if desired,
    # or just let it print.
    accuracy = wle.evaluate_model_word_level(
        test_df, model, grapheme_encoder, phoneme_encoder
    )
    return accuracy

def full_evaluation(model, X_test, y_test, test_df, g_enc, p_enc, model_name):
    """
    Runs all evaluations and returns a dictionary of results.
    """
    print(f"Evaluating {model_name}...")
    
    # 1. Letter Accuracy
    l_acc, _ = calculate_letter_accuracy(model, X_test, y_test)
    print(f"  > Letter Accuracy: {l_acc:.4f}")
    
    # 2. Word Accuracy
    # Note: We only run this on a sample if the dataset is huge, 
    # but for typical CMU dict sizes (split), it's fine.
    w_acc = calculate_word_accuracy(test_df, model, g_enc, p_enc)
    
    return {
        'letter_accuracy': l_acc,
        'word_accuracy': w_acc
    }