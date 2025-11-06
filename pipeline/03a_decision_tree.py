'''
    Purpose: Extract features from data and train 
    a decision tree model
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def parse_alignment(alignment):
    '''
    parse aligned grapheme-phoneme data
    in the form [(g1, p1), (g2, p2)]
    '''
    try:
        pairs = eval(alignment)
        # need null phoneme
        return [(g, p) for g, p in pairs] 
    except:
        return []

def extract_features(data):
    '''
    1. Extract context window as features from aligned data
    2. Save features and target as a JSON file
    3. Return DataFrame containing features and target
    '''   
    feature_target_pairs = []
    context_size = 2 # context window = +- 2 letters

    print('\nExtracting features...')
    for _, row in data.iterrows():
        alignment = parse_alignment(row['Alignment'])
        letters = [g for g, _ in alignment]
        phonemes = [p for _, p in alignment]

        for i in range(len(letters)):
            # extract context window, '_' if cpntext window out of bound
            left_context = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
            right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
            feature = left_context[::-1] + [letters[i]] + right_context
            feature_target_pairs.append({"features": feature, "target": phonemes[i]})

    print("Feature extraction done!")
    df_feature_target = pd.DataFrame(feature_target_pairs)
    output_path = "data/processed_data/decision_tree_feature_target.json"
    df_feature_target.to_json(output_path, orient="records", lines=True)
    print(f"Features and target saved to {output_path}")
    
    return df_feature_target

def encode_data(data: pd.DataFrame):
    '''
    Input: DataFrame containing extracted features and target
    Output: encoded features, target and encoder
    '''
    print("\nEncoding data for training...")
    
    letters = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    grapheme_encoder = LabelEncoder()
    grapheme_encoder.fit(letters)
    X_encoded_list = [grapheme_encoder.transform(f) for f in data["features"]]
    X = np.stack(X_encoded_list) # convert list of 1D arrays into one 2D array
    
    # encode phonemes
    phoneme_enc = LabelEncoder()
    y_enc = phoneme_enc.fit_transform(data["target"])

    print("Encoding done!")
    print(f"Sample original features: {data['features'][100]} -> Target: {data['target'][100]}")
    print("Encoded features:", X[100])
    print("Encoded target:", y_enc[100])
    return X, y_enc, grapheme_encoder, phoneme_enc

def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    '''
    1. Split data into training and testing sets.
    2. Train a decision tree classifier.
    3. Evaluate the model and print metrics.
    '''
    print("\n--- Training Decision Tree Model ---")
    
    # 1. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    
    # 2. Train the model
    # Use a basic Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    print("Model training complete.")
    
    # 3. Evaluate the model
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Evaluation Results ---")
    print(f"Model Test Accuracy: **{accuracy:.4f}**")
    
    return dt_classifier

def main():
    # load data
    path = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
    data = pd.read_csv(path)
    print(f"Loaded data from {path}")
  
    # extract feautres
    feature_target = extract_features(data)

    # encode features
    X, y, grapheme_encoder, phoneme_encoder = encode_data(feature_target)
    
    # train and evaluate
    decision_tree_model = train_and_evaluate(X, y)

if __name__ == "__main__":
    main()  