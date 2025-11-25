'''
Purpose: Centralized evaluation logic.
Calculates Per-Letter Accuracy and Per-Word Accuracy.
Updated to pass context_size AND measure Prediction Time.
'''

import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
import modules.word_evaluation as wle

def calculate_letter_accuracy(model, X_test, y_test):
    """
    Standard sklearn accuracy for isolated letter predictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred

def calculate_word_accuracy(test_df, model, grapheme_encoder, phoneme_encoder, context_size):
    """
    Uses word_evaluation.py to reconstruct full words and check correctness.
    """
    # Pass context_size to the lower module
    accuracy = wle.evaluate_model_word_level(
        test_df, model, grapheme_encoder, phoneme_encoder, context_size
    )
    return accuracy

def full_evaluation(model, X_test, y_test, test_df, g_enc, p_enc, model_name, context_size=2):
    """
    Runs all evaluations and returns a dictionary of results.
    NOW MEASURES PREDICTION TIME.
    """
    print(f"Evaluating {model_name}...")
    
    # --- START TIMER ---
    start_time = time.time()
    
    # 1. Letter Accuracy
    l_acc, _ = calculate_letter_accuracy(model, X_test, y_test)
    print(f"  > Letter Accuracy: {l_acc:.4f}")
    
    # 2. Word Accuracy
    w_acc = calculate_word_accuracy(test_df, model, g_enc, p_enc, context_size)
    
    # --- STOP TIMER ---
    end_time = time.time()
    eval_duration = end_time - start_time
    
    return {
        'letter_accuracy': l_acc,
        'word_accuracy': w_acc,
        'eval_time': eval_duration
    }