'''
    Purpose: Extract features from data and train 
    a Rule-Based + Machine Learning (Hybrid) model.

    This model uses a simple rule-based system for high-confidence
    predictions and falls back to a Decision Tree for all other cases.
'''

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Helper Functions (Identical to your DT script) ---

def parse_alignment(alignment):
    '''
    Parse aligned grapheme-phoneme data in the form [(g1, p1), (g2, p2)]
    '''
    try:
        pairs = eval(alignment)
        return [(g, p) for g, p in pairs] 
    except:
        return []

def extract_features(data):
    '''
    Extract context window as features from aligned data
    ''' 
    feature_target_pairs = []
    context_size = 2 # context window = +- 2 letters

    print('\nExtracting features...')
    for _, row in data.iterrows():
        alignment = parse_alignment(row['Alignment'])
        letters = [g for g, _ in alignment]
        phonemes = [p for _, p in alignment]

        for i in range(len(letters)):
            left_context = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
            right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
            feature = left_context[::-1] + [letters[i]] + right_context
            feature_target_pairs.append({"features": feature, "target": phonemes[i]})

    print("Feature extraction done!")
    df_feature_target = pd.DataFrame(feature_target_pairs)
    return df_feature_target

def encode_data(data: pd.DataFrame):
    '''
    Encode features and target for the Decision Tree
    '''
    print("\nEncoding data for training...")
    
    letters = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    grapheme_encoder = LabelEncoder()
    grapheme_encoder.fit(letters)
    X_encoded_list = [grapheme_encoder.transform(f) for f in data["features"]]
    X = np.stack(X_encoded_list)
    
    phoneme_enc = LabelEncoder()
    y_enc = phoneme_enc.fit_transform(data["target"])

    print("Encoding done!")
    return X, y_enc, grapheme_encoder, phoneme_enc

# --- NEW: Rule-Based System ---

def rule_based_predictor(features_raw):
    '''
    Applies a set of simple, high-confidence rules.
    If a rule matches, it returns the phoneme.
    Otherwise, it returns None.
    
    Features are [L2, L1, C, R1, R2]
    e.g., ['a', 'a', 'b', 'e', 'r']
    '''
    L1 = features_raw[1] # Letter to the left
    C = features_raw[2]  # Current letter
    R1 = features_raw[3] # Letter to the right

    # Rule 1: 'ch' -> 'CH'
    if C == 'c' and R1 == 'h':
        return 'CH'
    
    # Rule 2: 'sh' -> 'SH'
    if C == 's' and R1 == 'h':
        return 'SH'
        
    # Rule 3: 'th' -> 'TH'
    if C == 't' and R1 == 'h':
        return 'TH'

    # Rule 4: 'h' in 'ch', 'sh', 'th' is silent
    if C == 'h' and (L1 == 'c' or L1 == 's' or L1 == 't'):
        return '#' # Silent
        
    # Rule 5: Silent 'e' at the end of a word
    # (A simple version of the rule)
    if C == 'e' and R1 == '_':
        return '#' # Silent

    # No rules matched
    return None

# --- NEW: Hybrid Evaluation ---

def evaluate_hybrid_model(X_test_raw, X_test_enc, y_test, dt_model, phoneme_encoder):
    '''
    Iterates through the test set and applies the hybrid logic.
    Returns the list of final hybrid predictions.
    '''
    print("\nEvaluating hybrid model...")
    hybrid_predictions_enc = []
    
    # We need to transform these once to avoid errors
    known_phonemes = set(phoneme_encoder.classes_)
    
    rules_fired = 0
    ml_fired = 0
    
    for i in range(len(X_test_raw)):
        raw_features = X_test_raw[i]
        encoded_features = X_test_enc[i]
        
        # 1. Try the rules first
        rule_pred_str = rule_based_predictor(raw_features)
        
        if rule_pred_str is not None:
            # 2. Rule fired! Use its prediction.
            # We must check if the rule's output is a known phoneme
            if rule_pred_str in known_phonemes:
                pred_enc = phoneme_encoder.transform([rule_pred_str])[0]
                hybrid_predictions_enc.append(pred_enc)
                rules_fired += 1
            else:
                # Rule gave a phoneme we've never seen (e.g., 'CH' isn't in a small sample)
                # In this case, fall back to ML
                pred_enc = dt_model.predict([encoded_features])[0]
                hybrid_predictions_enc.append(pred_enc)
                ml_fired += 1
        
        else:
            # 3. Rule failed (returned None). Fall back to the ML model.
            pred_enc = dt_model.predict([encoded_features])[0]
            hybrid_predictions_enc.append(pred_enc)
            ml_fired += 1
            
    print(f"Evaluation complete. Rules used: {rules_fired}, ML fallback used: {ml_fired}")
    return hybrid_predictions_enc

# --- Main Execution ---

def main():
    # 1. Load data
    path = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
    data = pd.read_csv(path)
    print(f"Loaded data from {path}")
 
    # 2. Extract features
    feature_target = extract_features(data)

    # 3. Encode features (for the ML part)
    X_enc, y_enc, grapheme_encoder, phoneme_encoder = encode_data(feature_target)
    
    # 4. Get the raw (string) features for the Rule-Based part
    X_raw = feature_target['features'].tolist()

    # 5. Split ALL data (encoded X, raw X, and encoded y)
    # We must split them all together to keep them aligned
    print("\nSplitting data into 80/20 train/test sets...")
    X_train_enc, X_test_enc, X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_enc, X_raw, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Data split: Train samples={len(y_train)}, Test samples={len(y_test)}")
    
    # 6. Train the ML (Decision Tree) component
    print("\n--- Training ML (Decision Tree) Component ---")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_enc, y_train)
    print("ML component training complete.")
    
    # 7. Evaluate the ML component *by itself* for comparison
    ml_pred = dt_classifier.predict(X_test_enc)
    ml_accuracy = accuracy_score(y_test, ml_pred)
    print(f"\n--- ML Decision Tree (Baseline) ---")
    print(f"ML-Only Accuracy: **{ml_accuracy:.4f}**")
    
    # 8. Evaluate the Hybrid Model
    hybrid_predictions = evaluate_hybrid_model(
        X_test_raw, X_test_enc, y_test, dt_classifier, phoneme_encoder
    )
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
    
    # 9. Print final comparison
    print(f"\n--- Hybrid Model Evaluation ---")
    print(f"Hybrid Model Accuracy: **{hybrid_accuracy:.4f}**")
    
    print("\n--- Final Comparison ---")
    print(f"ML Decision Tree Accuracy: {ml_accuracy:.4f}")
    print(f"Hybrid Model Accuracy:     {hybrid_accuracy:.4f}")
    
    print("\n--- Hybrid Model Performance Table (Classification Report) ---")
    # Get all class names from the encoder
    class_names = phoneme_encoder.classes_
    print(classification_report(y_test, hybrid_predictions, labels=phoneme_encoder.transform(class_names), 
                                target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()