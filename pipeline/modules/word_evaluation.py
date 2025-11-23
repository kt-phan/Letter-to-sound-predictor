'''
Purpose: Evaluate decision tree model at the word level
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import modules.feature_extraction as fe

def get_word_predictions(word_alignment, model, grapheme_encoder, phoneme_encoder):
    pairs = fe.parse_alignment(word_alignment)
    letters = [g for g, _ in pairs]
    
    predicted_phonemes = []
    context_size = 2
    
    for i in range(len(letters)):
        # Extract features
        left_context = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
        right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
        feature = left_context[::-1] + [letters[i]] + right_context
        
        # Encode using fe module
        feature_encoded = fe.encode_features(pd.Series([feature]), grapheme_encoder)[0]
        
        # Predict
        pred_enc = model.predict([feature_encoded])[0]
        pred_phoneme = phoneme_encoder.inverse_transform([pred_enc])[0]
        
        if pred_phoneme != '#':  # Filter out null phonemes
            predicted_phonemes.append(pred_phoneme)
    
    return ' '.join(predicted_phonemes)

def word_level_accuracy(test_df, model, grapheme_encoder, phoneme_encoder):
   
    correct_words = 0
    total_words = len(test_df)
    
    debug_count = 0
    
    for _, row in test_df.iterrows():
        actual_phonemes = row['Pronunciation']  
        predicted_phonemes = get_word_predictions(row['Alignment'], model, grapheme_encoder, phoneme_encoder)
        
        # DEBUG: Print first 10 mismatches
        if debug_count < 10:
            match = "T" if actual_phonemes == predicted_phonemes else "F"
            print(f"\n{match} Word: {row.get('Word', 'N/A')}")
            print(f"  Actual:    '{actual_phonemes}'")
            print(f"  Predicted: '{predicted_phonemes}'")
            if actual_phonemes != predicted_phonemes:
                debug_count += 1
        
        if actual_phonemes == predicted_phonemes:
            correct_words += 1
    
    accuracy = correct_words / total_words
    return accuracy, correct_words, total_words

def evaluate_model_word_level(test_df, model, grapheme_encoder, phoneme_encoder):

    print("\n--- Word-Level Evaluation ---")
    
    accuracy, correct, total = word_level_accuracy(
        test_df, model, grapheme_encoder, phoneme_encoder
    )
    
    print(f"Words Correct: {correct}/{total}")
    print(f"Word-Level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy