'''
    Purpose: Extract features from data, train a K-Nearest Neighbors (KNN) model,
    and generate performance reports, tables, and plots.
'''

import pandas as pd
import numpy as np
import time
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
    1. Extract context window as features from aligned data
    2. Return DataFrame containing features and target
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

    print(f"Feature extraction done! Generated {len(feature_target_pairs)} examples.")
    df_feature_target = pd.DataFrame(feature_target_pairs)
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
    X_features = data["features"].tolist()
    
    phoneme_enc = LabelEncoder()
    y_enc = phoneme_enc.fit_transform(data["target"])

    print("Encoding done!")
    return X_features, y_enc, phoneme_enc

def train_and_evaluate_knn(X: list, y: np.ndarray):
    '''
    1. Split data, create Pipeline, train KNN, and evaluate.
    2. Returns history for plotting.
    '''
    print("\n--- Training K-Nearest Neighbors (KNN) Model ---")
    
    # 1. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    
    # 2. Create the pipeline
    knn_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))
    ])
    
    # 3. Train the model
    print("Training KNN pipeline...")
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
    
    # 5. Return results needed for plotting
    return y_test, y_pred, accuracy

# --- NEW: FUNCTIONS FOR PLOTS AND TABLES ---

def generate_performance_report(y_test, y_pred, encoder: LabelEncoder):
    '''
    Generates a classification report and plots for model performance.
    '''
    print("\n--- Performance Analysis ---")
    
    # 1. Generate the Classification Report (The Table)
    print("Classification Report (Table):")
    # Get all class names from the encoder
    class_names = encoder.classes_
    print(classification_report(y_test, y_pred, labels=encoder.transform(class_names), 
                                target_names=class_names, zero_division=0))

    # 2. Generate Confusion Matrix Plot
    print("Generating confusion matrix plot...")
    
    # Get all class names
    all_class_names = encoder.classes_
    
    # Find the top 20 most frequent classes in the test set
    unique, counts = np.unique(y_test, return_counts=True)
    top_indices = unique[np.argsort(-counts)][:20]
    top_class_names = encoder.inverse_transform(top_indices)

    # Plot and save a confusion matrix for *only* these top 20 classes
    fig, ax = plt.subplots(figsize=(15, 15))
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        labels=top_indices, 
        display_labels=top_class_names, 
        ax=ax,
        normalize='true',
        cmap='Blues',
        xticks_rotation='vertical'
    )
    ax.set_title("KNN Confusion Matrix (Top 20 Phonemes)")
    plt.tight_layout()
    plt.savefig('knn_confusion_matrix.png')
    print("Saved 'knn_confusion_matrix.png'")

    # 3. Generate Per-Class Accuracy Bar Chart
    print("Generating per-class accuracy plot...")
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_test, y_pred, labels=top_indices, normalize='true')
    per_class_accuracy = cm.diagonal()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(top_class_names, per_class_accuracy)
    ax.set_title('KNN Per-Class Accuracy (Top 20 Phonemes)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Phoneme')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig('knn_top_phoneme_accuracy.png')
    print("Saved 'knn_top_phoneme_accuracy.png'")

# --- END OF NEW FUNCTIONS ---

def main():
    # 1. Load data
    path = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at '{path}'")
        return
    print(f"Loaded data from {path} (Rows: {len(data)})")
 
    # 2. Extract features
    feature_target = extract_features(data)

    # 3. Sample the data for practical speed
    sample_size = 100000
    if len(feature_target) > sample_size:
        print(f"\nSampling down to {sample_size} rows for faster KNN training...")
        feature_target = feature_target.sample(n=sample_size, random_state=42)

    # 4. Encode features for KNN
    X_features, y_enc, phoneme_encoder = encode_data_for_knn(feature_target)
    
    # 5. Train and evaluate
    y_test, y_pred, accuracy = train_and_evaluate_knn(X_features, y_enc)
    
    # 6. --- Generate all reports and plots ---
    if y_test is not None:
        # We need the original encoder to get the phoneme names back
        generate_performance_report(y_test, y_pred, phoneme_encoder)

if __name__ == "__main__":
    main()