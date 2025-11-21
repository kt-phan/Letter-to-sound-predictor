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
import word_evaluation as wle


def main():
    
  # --- 1. Load and Extract Raw Data ---
  raw_df = pd.read_csv("data/processed_data/cmu_dict_cleaned_filtered_aligned.csv")

  # --- 2. Split Data  ---
  train_words, test_words = train_test_split(
      raw_df, 
      test_size=0.2, 
      random_state=42
  )


  # --- 3. Define X and y (Raw) ---
  train_features_df = fe.extract_features(train_words)
  test_features_df = fe.extract_features(test_words)
  X_train_raw = train_features_df['features']
  y_train = train_features_df['target']
  X_test_raw = test_features_df['features']
  y_test = test_features_df['target']

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

  # Phoneme-level accuracy
  accuracy = accuracy_score(y_test_enc, y_pred)
  print(f"Phoneme-Level Accuracy: **{accuracy:.4f}**")

  from sklearn.metrics import classification_report

  print(classification_report(y_test_enc, y_pred))


    # Word-level accuracy
  wle.evaluate_model_word_level(
        test_words,  # ← CHANGED: Pass the actual test words dataframe
        dt_classifier, 
        grapheme_encoder, 
        phoneme_encoder
    )
if __name__ == "__main__":
    main()  