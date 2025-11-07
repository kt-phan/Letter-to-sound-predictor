'''
    Purpose: Extract features from data and train 
    a K-Nearest Neighbors (KNN) model.

    This script is the KNN equivalent of the Decision Tree script.
    Key differences:
    1.  Uses OneHotEncoder (inside a Pipeline) for features, not LabelEncoder.
    2.  Uses a Pipeline to chain the encoder and the classifier.
    3.  Uses KNeighborsClassifier instead of DecisionTreeClassifier.
'''

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def parse_alignment(alignment):
    '''
    parse aligned grapheme-phoneme data
    in the form [(g1, p1), (g2, p2)]
    (Identical to your Decision Tree script)
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
    (Identical to your Decision Tree script, but saves a new file)
    ''' 
    feature_target_pairs = []
    context_size = 2 # context window = +- 2 letters

    print('\nExtracting features...')
    for _, row in data.iterrows():
        alignment = parse_alignment(row['Alignment'])
        letters = [g for g, _ in alignment]
        phonemes = [p for _, p in alignment]

        for i in range(len(letters)):
            # extract context window, '_' if context window out of bound
            left_context = [letters[i - j - 1] if i - j - 1 >= 0 else "_" for j in range(context_size)]
            right_context = [letters[i + j + 1] if i + j + 1 < len(letters) else "_" for j in range(context_size)]
            
            # The features 'X' are now a list of strings
            feature = left_context[::-1] + [letters[i]] + right_context
            
            # The target 'y' is the single corresponding phoneme
            feature_target_pairs.append({"features": feature, "target": phonemes[i]})

    print("Feature extraction done!")
    df_feature_target = pd.DataFrame(feature_target_pairs)
    
    # Save to a new JSON file to avoid overwriting your DT data
    output_path = "data/processed_data/knn_feature_target.json"
    df_feature_target.to_json(output_path, orient="records", lines=True)
    print(f"Features and target saved to {output_path}")
    
    return df_feature_target

def encode_data_for_knn(data: pd.DataFrame):
    '''
    Input: DataFrame containing extracted features and target
    Output:
        X_features: The raw features (list of lists of strings)
        y_enc: The LabelEncoded target
        phoneme_encoder: The fitted encoder for the target
    '''
    print("\nEncoding data for KNN...")
    
    # *** KEY DIFFERENCE FOR KNN ***
    # We keep X_features as their original categories (lists of strings).
    # The OneHotEncoder in our Pipeline will handle them at train time.
    X_features = data["features"].tolist()
    
    # We ONLY need to LabelEncode the target variable 'y'
    phoneme_enc = LabelEncoder()
    y_enc = phoneme_enc.fit_transform(data["target"])

    print("Encoding done!")
    print(f"Sample original features: {X_features[100]} -> Target: {data['target'][100]}")
    print("Encoded target:", y_enc[100])
    
    return X_features, y_enc, phoneme_enc

def train_and_evaluate_knn(X: list, y: np.ndarray):
    '''
    1. Split data into training and testing sets.
    2. Create a Pipeline with OneHotEncoder and KNeighborsClassifier.
    3. Train the KNN model.
    4. Evaluate the model and print metrics.
    '''
    print("\n--- Training K-Nearest Neighbors (KNN) Model ---")
    
    # 1. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    
    # 2. Create the pipeline
    # This is the most important part for KNN.
    # Step 1: 'encoder' - Converts categorical features (letters) into a numerical vector
    # Step 2: 'knn' - The actual classifier
    knn_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))
    ])
    # n_neighbors=5 is a good default. 'weights=distance' is often better than default.
    # n_jobs=-1 uses all your CPU cores to speed up prediction.
    
    # 3. Train the model
    print("Training KNN pipeline... (this may take a few minutes)")
    start_time = time.time()
    knn_pipeline.fit(X_train, y_train)
    print(f"Model training complete. Time: {time.time() - start_time:.2f}s")
    
    # 4. Evaluate the model
    print("Evaluating model...")
    y_pred = knn_pipeline.predict(X_test)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Evaluation Results ---")
    print(f"Model Test Accuracy: **{accuracy:.4f}**")
    
    return knn_pipeline

def main():
    # load data
    # --- Using the correct project path from your code ---
    path = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
    
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at '{path}'")
        print("Please make sure the file is in your project's 'data/processed_data' folder.")
        return
        
    print(f"Loaded data from {path}")
 
    # extract feautres
    feature_target = extract_features(data)

    # encode features for KNN
    X_features, y_enc, phoneme_encoder = encode_data_for_knn(feature_target)
    
    # train and evaluate
    knn_model = train_and_evaluate_knn(X_features, y_enc)

if __name__ == "__main__":
    main()