'''
Purpose: Evaluate decision tree model at the word level
Updated to accept dynamic context_size.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import modules.feature_extraction as fe

def get_word_predictions(word_alignment, model, grapheme_encoder, phoneme_encoder, context_size):
    pairs = fe.parse_alignment(word_alignment)
    letters = [g for g, _ in pairs]
    
    predicted_phonemes = []
    # context_size is now passed in as an argument
    
    for i in range(len(letters)):
        # Extract features using the correct context size
        left_context = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
        right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
        
        # Note: Ensure reversal logic matches feature_extraction.py
        feature = left_context[::-1] + [letters[i]] + right_context
        
        # Encode using fe module
        feature_encoded = fe.encode_features(pd.Series([feature]), grapheme_encoder)[0]
        
        # Predict
        pred_enc = model.predict([feature_encoded])[0]
        pred_phoneme = phoneme_encoder.inverse_transform([pred_enc])[0]
        
        if pred_phoneme != '#':  # Filter out null phonemes
            predicted_phonemes.append(pred_phoneme)
    
    return ' '.join(predicted_phonemes)

def word_level_accuracy(test_df, model, grapheme_encoder, phoneme_encoder, context_size):
   
    correct_words = 0
    total_words = len(test_df)
    
    debug_count = 0
    
    for _, row in test_df.iterrows():
        actual_phonemes = row['Pronunciation']  
        # Pass context_size down
        predicted_phonemes = get_word_predictions(row['Alignment'], model, grapheme_encoder, phoneme_encoder, context_size)
        
        # DEBUG: Print first 10 mismatches
        if debug_count < 10:
            if actual_phonemes != predicted_phonemes:
                # print(f"Miss: {row.get('Word')} -> Exp: {actual_phonemes} | Got: {predicted_phonemes}")
                debug_count += 1
        
        if actual_phonemes == predicted_phonemes:
            correct_words += 1
    
    accuracy = correct_words / total_words
    return accuracy, correct_words, total_words

def evaluate_model_word_level(test_df, model, grapheme_encoder, phoneme_encoder, context_size):

    print("\n--- Word-Level Evaluation ---")
    
    # Pass context_size down
    accuracy, correct, total = word_level_accuracy(
        test_df, model, grapheme_encoder, phoneme_encoder, context_size
    )
    
    print(f"Words Correct: {correct}/{total}")
    print(f"Word-Level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy