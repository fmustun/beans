import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Watkins dolphin species labels
LABELS = [
    "Clymene_Dolphin",
    "Bottlenose_Dolphin", 
    "Spinner_Dolphin",
    "White-sided_Dolphin",
    "White-beaked_Dolphin",
    "Frasers_Dolphin",
    "Atlantic_Spotted_Dolphin",
    "Rough-Toothed_Dolphin",
    "Pantropical_Spotted_Dolphin",
    "Striped_Dolphin",
    "Common_Dolphin"
]

def read_confusion_matrix(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find the confusion matrix section
    matrix_match = re.search(r'Test confusion matrix:\n(.*?)\nvalid_metric_best', content, re.DOTALL)
    if not matrix_match:
        raise ValueError("Could not find confusion matrix in log file")
    
    # Parse the matrix
    matrix_str = matrix_match.group(1).strip()
    matrix = []
    for line in matrix_str.split('\n'):
        # Remove all square brackets and split by whitespace
        row = [int(x) for x in re.sub(r'[\[\]]', '', line).split()]
        matrix.append(row)
    
    return np.array(matrix)

def plot_confusion_matrix(matrix, labels, output_file=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', required=True, help='Path to the log file containing the confusion matrix')
    parser.add_argument('--output', help='Path to save the plot (optional)')
    args = parser.parse_args()
    
    matrix = read_confusion_matrix(args.log_file)
    plot_confusion_matrix(matrix, LABELS, args.output) 