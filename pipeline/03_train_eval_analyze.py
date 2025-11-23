'''
Purpose: Master training script. 
Trains Decision Tree, KNN, Hybrid, and HMM models.
'''

import pandas as pd
import numpy as np
import time
import ast
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Custom Modules
import modules.feature_extraction as fe
import modules.evaluate_models as em
import modules.analysis as an

# --- Configuration ---
# We evaluate ALL models on this same subset of words to ensure:
# 1. Fairness (Apples-to-apples comparison)
# 2. Speed (KNN prediction on full test set is too slow)
EVAL_SAMPLE_SIZE = 1000 

# --- Hybrid Model Wrapper ---
class HybridModel:
    def __init__(self, dt_model, grapheme_encoder, phoneme_encoder):
        self.dt_model = dt_model
        self.grapheme_encoder = grapheme_encoder
        self.phoneme_encoder = phoneme_encoder
        self.known_phonemes = set(phoneme_encoder.classes_)
        
        self.boundary = grapheme_encoder.transform(['_'])[0]
        self.c = grapheme_encoder.transform(['c'])[0]
        self.s = grapheme_encoder.transform(['s'])[0]
        self.t = grapheme_encoder.transform(['t'])[0]
        self.h = grapheme_encoder.transform(['h'])[0]
        self.e = grapheme_encoder.transform(['e'])[0]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(1, -1)
        dt_preds = self.dt_model.predict(X)
        final_predictions = []
        for i, features in enumerate(X):
            L1, C, R1 = features[1], features[2], features[3]
            rule_phoneme = None
            if R1 == self.h:
                if C == self.c: rule_phoneme = 'CH'
                elif C == self.s: rule_phoneme = 'SH'
                elif C == self.t: rule_phoneme = 'TH'
            if C == self.h and L1 in [self.c, self.s, self.t]: rule_phoneme = '#'
            if C == self.e and R1 == self.boundary: rule_phoneme = '#'

            if rule_phoneme and rule_phoneme in self.known_phonemes:
                pred = self.phoneme_encoder.transform([rule_phoneme])[0]
                final_predictions.append(pred)
            else:
                final_predictions.append(dt_preds[i])
        return np.array(final_predictions)

# --- HMM / Viterbi Model Wrapper ---
class HMMModel:
    def __init__(self, dt_model, grapheme_encoder, phoneme_encoder):
        self.dt_model = dt_model
        self.grapheme_encoder = grapheme_encoder
        self.phoneme_encoder = phoneme_encoder
        self.phoneme_map = {i: p for i, p in enumerate(phoneme_encoder.classes_)}
        self.boundary_marker = '_'

    def predict(self, X):
        return self.dt_model.predict(X)

    def predict_word(self, alignment_str):
        try:
            pairs = ast.literal_eval(alignment_str)
            letters = [g for g, _ in pairs]
        except:
            return ""

        if not letters: return ""

        padded = [self.boundary_marker]*2 + letters + [self.boundary_marker]*2
        features_list = []
        for i in range(len(letters)):
            window = padded[i : i + 5] 
            features_list.append(window)
        
        X_encoded = fe.encode_features(pd.Series(features_list), self.grapheme_encoder)
        emission_probs = self.dt_model.predict_proba(X_encoded)
        
        T, N = emission_probs.shape
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        delta[0, :] = emission_probs[0, :]
        
        for t in range(1, T):
            for j in range(N):
                path_probs = delta[t-1, :] * emission_probs[t, j]
                delta[t, j] = np.max(path_probs)
                psi[t, j] = np.argmax(path_probs)

        best_last_state = np.argmax(delta[T-1, :])
        path = [0] * T
        path[T-1] = best_last_state
        for t in range(T - 2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        decoded = [self.phoneme_map.get(idx, '#') for idx in path]
        decoded_clean = [p for p in decoded if p != '#']
        
        return ' '.join(decoded_clean)

def main():
    # 1. Load Data
    print("Loading data...")
    path = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
    try:
        raw_df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return
        
    # 2. Split Data (WORDS)
    train_words_df, test_words_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    print(f"Total Split: Train Words {len(train_words_df)}, Test Words {len(test_words_df)}")

    # --- Common Evaluation Subset ---
    if len(test_words_df) > EVAL_SAMPLE_SIZE:
        print(f"Creating common evaluation set of {EVAL_SAMPLE_SIZE} words...")
        eval_words_df = test_words_df.iloc[:EVAL_SAMPLE_SIZE].copy()
    else:
        eval_words_df = test_words_df.copy()

    # 3. Extract Features
    print("Extracting features...")
    train_feat_df = fe.extract_features(train_words_df)
    eval_feat_df = fe.extract_features(eval_words_df)
    
    # 4. Encoders
    print("Fitting encoders...")
    grapheme_encoder = fe.get_grapheme_encoder()
    phoneme_encoder = fe.fit_phoneme_encoder(train_feat_df['target'])
    
    # 5. Prepare Matrices
    X_train = fe.encode_features(train_feat_df['features'], grapheme_encoder)
    y_train = fe.encode_target(train_feat_df['target'], phoneme_encoder)
    X_test = fe.encode_features(eval_feat_df['features'], grapheme_encoder)
    y_test = fe.encode_target(eval_feat_df['target'], phoneme_encoder)

    model_results = {}

    # --- Model 1: Decision Tree ---
    print("\n" + "-"*30)
    print("Model 1: Decision Tree")
    start = time.time()
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    
    metrics = em.full_evaluation(dt, X_test, y_test, eval_words_df, grapheme_encoder, phoneme_encoder, "Decision Tree")
    metrics['time'] = train_time
    model_results['Decision Tree'] = metrics

    # --- Model 2: KNN ---
    print("\n" + "-"*30)
    print("Model 2: KNN (Enabled)")
    knn_train_limit = 50000
    if len(X_train) > knn_train_limit:
        print(f"  (Subsampling KNN training data to {knn_train_limit} letters for speed)")
        X_knn_train = X_train[:knn_train_limit]
        y_knn_train = y_train[:knn_train_limit]
    else:
        X_knn_train, y_knn_train = X_train, y_train

    start = time.time()
    # Use OneHotEncoder so KNN sees categories, not ordinal integers
    knn = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))
    ])
    knn.fit(X_knn_train, y_knn_train)
    train_time = time.time() - start

    metrics = em.full_evaluation(knn, X_test, y_test, eval_words_df, grapheme_encoder, phoneme_encoder, "KNN")
    metrics['time'] = train_time
    model_results['KNN'] = metrics

    # --- Model 3: Hybrid ---
    print("\n" + "-"*30)
    print("Model 3: Hybrid (Rule-Based)")
    hybrid = HybridModel(dt, grapheme_encoder, phoneme_encoder)
    metrics = em.full_evaluation(hybrid, X_test, y_test, eval_words_df, grapheme_encoder, phoneme_encoder, "Hybrid")
    metrics['time'] = 0 
    model_results['Hybrid'] = metrics

    # --- Model 4: HMM / Viterbi ---
    print("\n" + "-"*30)
    print("Model 4: HMM (Viterbi Decoder)")
    hmm = HMMModel(dt, grapheme_encoder, phoneme_encoder)
    metrics = em.full_evaluation(hmm, X_test, y_test, eval_words_df, grapheme_encoder, phoneme_encoder, "HMM")
    metrics['time'] = 0
    model_results['HMM'] = metrics

    # --- 6. Analysis ---
    print("\n" + "="*40)
    print("COMPARATIVE ANALYSIS")
    print("="*40)
    
    an.plot_model_comparison(model_results)
    
    # Deep dive into the best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['word_accuracy'])
    print(f"\nPerforming deep dive on best model: {best_model_name}...")
    
    if best_model_name == 'Decision Tree': model = dt
    elif best_model_name == 'KNN': model = knn
    elif best_model_name == 'Hybrid': model = hybrid
    else: model = hmm
    
    y_pred_best = model.predict(X_test)
    an.analyze_phoneme_performance(y_test, y_pred_best, phoneme_encoder)
    
    # Analyze Word Errors (KNN is INCLUDED now)
    an.analyze_word_errors(
        eval_words_df, 
        [dt, knn, hybrid, hmm],          
        ['DT', 'KNN', 'Hyb', 'HMM'],       
        grapheme_encoder, 
        phoneme_encoder
    )

    # --- NEW: Plot Accuracy by Word Length (KNN Included) ---
    an.plot_accuracy_by_word_length(
        eval_words_df,
        [dt, knn, hybrid, hmm],
        ['DT', 'KNN', 'Hyb', 'HMM'],
        grapheme_encoder,
        phoneme_encoder
    )

if __name__ == "__main__":
    main()