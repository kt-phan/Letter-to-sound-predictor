'''
Purpose: Post-training analysis. 
Generates plots for:
1. Model Comparison (Bar Chart)
2. Phoneme Performance (Horizontal Bar Chart)
3. Accuracy by Word Length (Line Chart)
4. Hardest Words List
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import modules.word_evaluation as wle

def plot_model_comparison(results_dict):
    """
    Plots Bar chart for Accuracy and Line chart for Time.
    """
    print("Generating Model Comparison plot...")
    models = list(results_dict.keys())
    l_accs = [results_dict[m]['letter_accuracy'] for m in models]
    w_accs = [results_dict[m]['word_accuracy'] for m in models]
    times = [results_dict[m]['time'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, l_accs, width, label='Letter Acc', color='skyblue')
    ax1.bar(x + width/2, w_accs, width, label='Word Acc', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Model Accuracy Comparison')
    ax1.legend()
    
    # Plot 2: Training Time
    ax2.plot(models, times, marker='o', color='red', linestyle='--')
    ax2.set_title('Training Time (seconds)')
    ax2.set_ylabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('data/analysis_output/analysis_model_comparison.png')
    print("Saved 'data/analysis_output/analysis_model_comparison.png'")


def analyze_phoneme_performance(y_true, y_pred, phoneme_encoder):
    """
    Plots the Top 10 and Bottom 10 phonemes by F1-Score.
    """
    print("\n--- Phoneme Performance Analysis ---")
    labels = phoneme_encoder.classes_
    
    # Generate report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(report).transpose()
    df_rep = df_rep.iloc[:-3] # Remove 'accuracy', 'macro avg', etc.
    
    # Save CSV
    df_rep.to_csv("data/analysis_output/analysis_phoneme_metrics.csv")
    
    # --- NEW: PLOT BEST AND WORST PHONEMES ---
    df_sorted = df_rep.sort_values(by='f1-score', ascending=True) # Sort by F1
    
    # Get Bottom 10 and Top 10
    bottom_10 = df_sorted.head(10)
    top_10 = df_sorted.tail(10)
    
    # Combine for plotting
    plot_df = pd.concat([bottom_10, top_10])
    
    plt.figure(figsize=(10, 8))
    # Color code: Red for bad, Green for good
    colors = ['salmon'] * len(bottom_10) + ['lightgreen'] * len(top_10)
    
    plt.barh(plot_df.index, plot_df['f1-score'], color=colors)
    plt.title('Phoneme Prediction Accuracy (F1-Score)')
    plt.xlabel('F1 Score (Higher is Better)')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('data/analysis_output/analysis_phoneme_performance.png')
    print("Saved 'data/analysis_output/analysis_phoneme_performance.png' (Visualizes hard vs. easy sounds)")


def plot_accuracy_by_word_length(test_df, models, model_names, g_enc, p_enc):
    """
    NEW: Plots a line chart showing how accuracy changes as words get longer.
    Useful for seeing if HMM/Hybrid handles long words better than DT.
    """
    print("\n--- Generating Accuracy by Word Length Plot ---")
    
    # Prepare data structure
    # length_stats = { length: { 'total': 0, 'DT_correct': 0, ... } }
    length_stats = {}
    
    # We iterate through the test set (this uses the Eval Subset passed from train_models)
    for _, row in test_df.iterrows():
        word_len = len(row['Word'])
        target = row['Pronunciation']
        alignment = row['Alignment']
        
        if word_len not in length_stats:
            length_stats[word_len] = {'total': 0}
            for name in model_names:
                length_stats[word_len][name] = 0
        
        length_stats[word_len]['total'] += 1
        
        # Check each model
        for model, name in zip(models, model_names):
            pred = wle.get_word_predictions(alignment, model, g_enc, p_enc)
            if pred == target:
                length_stats[word_len][name] += 1

    # Convert to DataFrame for plotting
    lengths = sorted(length_stats.keys())
    # Filter out lengths with too few samples (noise)
    valid_lengths = [l for l in lengths if length_stats[l]['total'] >= 5]
    
    plt.figure(figsize=(10, 6))
    
    for name in model_names:
        accuracies = []
        for l in valid_lengths:
            acc = length_stats[l][name] / length_stats[l]['total']
            accuracies.append(acc)
        
        plt.plot(valid_lengths, accuracies, marker='o', label=name)

    plt.title('Word Accuracy vs. Word Length')
    plt.xlabel('Word Length (Letters)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/analysis_output/analysis_accuracy_by_length.png')
    print("Saved 'data/analysis_output/analysis_accuracy_by_length.png'")


def analyze_word_errors(test_df, models, model_names, g_enc, p_enc):
    """
    Finds 'Hardest Words': Words that ALL models got wrong.
    """
    print("\n--- Word Error Analysis ---")
    
    results = []
    # Use the provided subset
    for _, row in test_df.iterrows():
        word = row['Word']
        target = row['Pronunciation']
        alignment = row['Alignment']
        
        entry = {'Word': word, 'Target': target}
        failures = 0
        
        for model, name in zip(models, model_names):
            pred = wle.get_word_predictions(alignment, model, g_enc, p_enc)
            entry[f'Pred_{name}'] = pred
            if pred != target:
                failures += 1
        
        entry['Failures'] = failures
        results.append(entry)
        
    results_df = pd.DataFrame(results)
    
    # Filter for words where Failures == len(models) (All models got it wrong)
    hardest_words = results_df[results_df['Failures'] == len(models)]
    
    print(f"Found {len(hardest_words)} words (in sample) that NO model predicted correctly.")
    hardest_words.to_csv("data/analysis_output/analysis_hardest_words.csv", index=False)
    print("Saved list of hardest words to 'data/analysis_output/analysis_hardest_words.csv'")