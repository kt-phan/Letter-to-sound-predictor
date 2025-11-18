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

import feature_extraction as fe

def main():
    
  # --- 1. Load and Extract Raw Data ---
  raw_df = pd.read_csv("data/processed_data/cmu_dict_cleaned_filtered_aligned.csv")
  data_df = fe.extract_features(raw_df)

  # --- 2. Define X and y (Raw) ---
  X_features = data_df['features'] # This is a Series of lists of strings
  y_target = data_df['target']   # This is a Series of phoneme strings

  # --- 3. Split Data  ---
  X_train_raw, X_test_raw, y_train, y_test = train_test_split(
      X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
  )
  print(f"Data split: Train samples={len(y_train)}, Test samples={len(y_test)}")

  # --- 4. Encode Data for Decision Tree ---
  # A) Get the pre-fitted grapheme "key"
  grapheme_encoder = fe.get_grapheme_encoder()

  # B) Encode X features using the key
  print("Encoding features for Decision Tree...")
  X_train_enc = fe.encode_features(X_train_raw, grapheme_encoder)
  X_test_enc = fe.encode_features(X_test_raw, grapheme_encoder)

  # C) Create, fit, and transform the Y (target) encoder
  print("Encoding target...")
  phoneme_encoder = fe.fit_phoneme_encoder(y_train)
  y_train_enc = fe.encode_target(y_train, phoneme_encoder)
  y_test_enc = fe.encode_target(y_test, phoneme_encoder)

  # --- 5. Train Model ---
  print("\n--- Training Decision Tree Model ---")
  dt_classifier = DecisionTreeClassifier(random_state=42)
  dt_classifier.fit(X_train_enc, y_train_enc)
  print("Model training complete.")

  # --- 6. Evaluate Model --
  y_pred = dt_classifier.predict(X_test_enc)
  accuracy = accuracy_score(y_test_enc, y_pred)
  print(f"Model Test Accuracy: **{accuracy:.4f}**")

if __name__ == "__main__":
    main()  