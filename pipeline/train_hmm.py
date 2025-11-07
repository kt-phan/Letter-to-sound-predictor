import pandas as pd
import numpy as np
import ast  # For safely parsing the alignment string
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

# --- 0. Configuration ---
warnings.filterwarnings('ignore', category=UserWarning) # Suppress warnings from sklearn

# --- 1. Load Data ---
file_name = 'data/processed_data/cmu_dict_cleaned_filtered_aligned.csv'
print(f"Loading data from {file_name}...")
try:
    df = pd.read_csv(file_name)
    print("Data loaded successfully.")
    print(f"Total entries: {len(df)}")
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

# --- 2. Feature Engineering: Create Context Window Dataset ---
print("Starting feature engineering (this may take a minute)...")
context_data = []
boundary_marker = '_' # Used for padding context at word boundaries

# Iterate over each word and its alignment
for _, row in df.iterrows():
    word = str(row['Word'])
    
    try:
        # Safely parse the string representation of the list of tuples
        # e.g., "[('a', '#'), ('a', 'AA')]" -> [('a', '#'), ('a', 'AA')]
        alignment = ast.literal_eval(row['Alignment'])
    except (ValueError, SyntaxError):
        # Skip rows with malformed alignment strings
        continue
        
    # Create a padded list of graphemes for context lookup
    graphemes = [pair[0] for pair in alignment]
    phonemes = [pair[1] for pair in alignment]
    
    # Pad the grapheme list for the +/- 2 context window
    # Example: 'cat' -> ['_', '_', 'c', 'a', 't', '_', '_']
    padded_graphemes = [boundary_marker] * 2 + graphemes + [boundary_marker] * 2
    
    for i in range(len(graphemes)):
        t = i + 2 # Current position in padded list
        
        # Extract the 5-grapheme context window
        g_t_minus_2 = padded_graphemes[t - 2]
        g_t_minus_1 = padded_graphemes[t - 1]
        g_t = padded_graphemes[t] # This is graphemes[i]
        g_t_plus_1 = padded_graphemes[t + 1]
        g_t_plus_2 = padded_graphemes[t + 2]
        
        # Target phoneme
        phoneme_t = phonemes[i]
        
        # Add the feature vector (X) and target (y) to our list
        context_data.append([g_t_minus_2, g_t_minus_1, g_t, g_t_plus_1, g_t_plus_2, phoneme_t])

# Create the feature DataFrame
print(f"Created {len(context_data)} letter-phoneme pairs with context.")
feature_df = pd.DataFrame(context_data, columns=['G_t-2', 'G_t-1', 'G_t', 'G_t+1', 'G_t+2', 'Phoneme'])

# --- 3. Create X and y, and Encode Features ---
print("Encoding features...")

X = feature_df[['G_t-2', 'G_t-1', 'G_t', 'G_t+1', 'G_t+2']]
y = feature_df['Phoneme']

# We need to encode all graphemes and phonemes consistently.
# Using LabelEncoder for y (phonemes)
phoneme_encoder = LabelEncoder()
y_encoded = phoneme_encoder.fit_transform(y)
# Store phoneme mappings for the decoder
phoneme_reverse_map = {i: c for i, c in enumerate(phoneme_encoder.classes_)}

# Using a manual mapping for X (graphemes) to handle all columns at once
all_graphemes = pd.unique(X.values.ravel('K')) # Find all unique graphemes
grapheme_map = {g: i for i, g in enumerate(all_graphemes)} # Create map
unknown_grapheme_code = grapheme_map[boundary_marker] # Code for unseen graphemes

X_encoded = X.apply(lambda col: col.map(grapheme_map))

# --- 4. Split Data ---
print("Splitting data into training and test sets (75/25)...")
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)

# --- 5. Train Decision Tree Classifier ---
print("Training Decision Tree classifier...")
# Limiting max_depth to 10 to prevent overfitting and speed up training
dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42) 
dt_classifier.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate Classifier ---
print("Evaluating classifier on test set...")
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 50)
print(f"Total Letter-Phoneme Pairs: {len(feature_df)}")
print(f"Training Pairs: {len(X_train)}, Test Pairs: {len(X_test)}")
print(f"Decision Tree Classifier Accuracy: {accuracy * 100:.2f}%")
print("-" * 50)


# --- 7. Viterbi Decoder (Illustrative Example) ---

def get_grapheme_features_for_word(word, grapheme_map, boundary_marker='_', unknown_code=0):
    """
    Transforms a single word into a sequence of 5-grapheme feature vectors
    and encodes them using the trained grapheme_map.
    """
    word_lower = word.lower()
    graphemes = list(word_lower)
    padded_graphemes = [boundary_marker] * 2 + graphemes + [boundary_marker] * 2
    
    features = []
    
    for i in range(len(graphemes)):
        t = i + 2
        context = [
            padded_graphemes[t - 2],
            padded_graphemes[t - 1],
            padded_graphemes[t],
            padded_graphemes[t + 1],
            padded_graphemes[t + 2]
        ]
        
        # Encode features using the trained grapheme_map
        # Use 'unknown_code' for any graphemes not seen during training
        encoded_features = [grapheme_map.get(g, unknown_code) for g in context]
        features.append(encoded_features)

    return np.array(features)

def hybrid_viterbi_decode(word, dt_model, grapheme_map, phoneme_reverse_map, boundary_marker='_'):
    """
    Performs Viterbi decoding using the Decision Tree's probability as the 
    Emission Score (B).
    """
    # 1. Get Context Features for the word
    unknown_code = grapheme_map.get(boundary_marker, 0)
    features = get_grapheme_features_for_word(word, grapheme_map, boundary_marker, unknown_code)
    T = len(features) # Length of the word
    
    if T == 0:
        return [], 0.0

    # 2. Get Emission Scores from the Decision Tree
    # This is the P(Phoneme | Context) for each letter
    emission_scores = dt_model.predict_proba(features)
    
    N = emission_scores.shape[1] # Number of possible phonemes
    
    # 3. Viterbi Algorithm
    delta = np.zeros((T, N)) # Stores max probability of path
    psi = np.zeros((T, N), dtype=int) # Stores backpointers

    # --- Initialization (t=0) ---
    # The path starts with the probabilities of the first letter
    delta[0, :] = emission_scores[0, :]
    
    # --- Recursion (t=1 to T-1) ---
    # We simplify this by assuming the transition probability (A) = 1.0 
    # (i.e., we are only maximizing the product of emission scores).
    # A more advanced model would include phonotactics (P(Phoneme_j | Phoneme_i)).
    
    for t in range(1, T):
        for j in range(N): # For each current phoneme j
            
            # Find the max probability path from ANY previous phoneme i
            # delta[t, j] = max_{i} [ delta[t-1, i] * A_{i,j} * B_{j}(o_t) ]
            # Here, A=1 and B is our emission_scores[t, j]
            
            path_probs = delta[t-1, :] * 1.0 * emission_scores[t, j] 
            
            delta[t, j] = np.max(path_probs)
            psi[t, j] = np.argmax(path_probs)

    # --- Termination and Backtracking ---
    best_path_prob = np.max(delta[T-1, :])
    best_last_state = np.argmax(delta[T-1, :])
    
    # Follow the backpointers to find the most likely sequence
    viterbi_path = [0] * T
    viterbi_path[T-1] = best_last_state
    
    for t in range(T - 2, -1, -1):
        viterbi_path[t] = psi[t+1, viterbi_path[t+1]]
        
    # Convert state indices back to phonemes
    decoded_phonemes = [phoneme_reverse_map.get(idx, 'UNK') for idx in viterbi_path]
    
    return decoded_phonemes, best_path_prob

# --- 8. Run Illustrative Example ---
print("Running illustrative example on the word 'washington'...")

test_word = "washington"
decoded_sequence, prob = hybrid_viterbi_decode(
    test_word, 
    dt_classifier, 
    grapheme_map, 
    phoneme_reverse_map, 
    boundary_marker
)

print(f"\nInput Word: {test_word}")
print(f"Decoded Phonemes: {' '.join(decoded_sequence)}")
print(f"(Note: The Viterbi path probability is {prob:.2e})")