import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_class_distribution(data, title, save_path=None, figsize=(10, 6)):
    """Plot class distribution with counts and percentages"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    class_counts = data['label'].value_counts()
    colors = sns.color_palette("husl", len(class_counts))
    
    bars1 = ax1.bar(class_counts.index, class_counts.values, color=colors)
    ax1.set_title(f'{title} - Class Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Class', fontsize=12)
    # Rotate x-axis labels to prevent overlapping
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add count labels on bars
    for bar, count in zip(bars1, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart with percentages
    percentages = (class_counts / len(data) * 100).round(1)
    wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title(f'{title} - Class Distribution (%)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()

def plot_dataset_comparison(original_data, filtered_data, train_data, valid_data, test_data, save_path=None):
    """Plot comparison of class distribution across different dataset stages"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    datasets = [
        (original_data, 'Original Dataset'),
        (filtered_data, 'After Duration Filter'),
        (train_data, 'Train Split'),
        (valid_data, 'Validation Split'),
        (test_data, 'Test Split')
    ]
    
    # Get all unique classes for consistent coloring
    all_classes = set()
    for data, _ in datasets:
        all_classes.update(data['label'].unique())
    all_classes = sorted(list(all_classes))
    colors = sns.color_palette("husl", len(all_classes))
    color_dict = dict(zip(all_classes, colors))
    
    for idx, (data, title) in enumerate(datasets):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        class_counts = data['label'].value_counts()
        class_colors = [color_dict[cls] for cls in class_counts.index]
        
        bars = ax.bar(class_counts.index, class_counts.values, color=class_colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        # Rotate x-axis labels to prevent overlapping
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add total count in title
        ax.set_title(f'{title}\n(Total: {len(data):,} samples)', fontsize=11, fontweight='bold')
    
    # Remove the last subplot if not needed
    if len(datasets) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    plt.show()

def plot_imbalance_metrics(data, title, save_path=None):
    """Plot imbalance metrics and statistics"""
    class_counts = data['label'].value_counts()
    total_samples = len(data)
    
    # Calculate imbalance metrics
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Imbalance Analysis: {title}', fontsize=16, fontweight='bold')
    
        # 1. Class distribution with imbalance ratio
    bars = ax1.bar(class_counts.index, class_counts.values, 
                   color=sns.color_palette("husl", len(class_counts)))
    ax1.set_title(f'Class Distribution\n(Imbalance Ratio: {imbalance_ratio:.2f}:1)', 
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    # Rotate x-axis labels to prevent overlapping
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Percentage distribution
    percentages = (class_counts / total_samples * 100).round(1)
    ax2.pie(class_counts.values, labels=[f'{cls}\n{pct}%' for cls, pct in zip(class_counts.index, percentages)],
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(class_counts)))
    ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')
    
    # 3. Log scale to better visualize imbalance
    ax3.bar(class_counts.index, class_counts.values, 
            color=sns.color_palette("husl", len(class_counts)))
    ax3.set_yscale('log')
    ax3.set_title('Class Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Samples (log scale)')
    # Rotate x-axis labels to prevent overlapping
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add count labels on log scale
    for i, (cls, count) in enumerate(class_counts.items()):
        ax3.text(i, count * 1.1, f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Imbalance statistics table
    ax4.axis('off')
    stats_text = f"""
    Dataset Statistics:
    
    Total Samples: {total_samples:,}
    Number of Classes: {len(class_counts)}
    
    Class Distribution:
    """
    for cls, count in class_counts.items():
        pct = (count / total_samples * 100)
        stats_text += f"\n{cls}: {count:,} ({pct:.1f}%)"
    
    stats_text += f"""
    
    Imbalance Metrics:
    - Most frequent class: {class_counts.index[0]} ({class_counts.iloc[0]:,} samples)
    - Least frequent class: {class_counts.index[-1]} ({class_counts.iloc[-1]:,} samples)
    - Imbalance ratio: {imbalance_ratio:.2f}:1
    - Standard deviation: {class_counts.std():.1f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved imbalance analysis to: {save_path}")
    
    plt.show()

def main():
    # Paths
    input_csv = 'data/dolphin_reef/all/all.csv'
    output_dir = 'data/dolphin_reef/unbalanced_SW_NSW/'
    plots_dir = 'Dolph2Vec_detector/plots/'
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading and processing dataset...")
    
    # Read original data
    original_csv = pd.read_csv(input_csv)
    print(f"Original dataset size: {len(original_csv):,} samples")
    
    # Apply the same processing as in the original script
    csv = original_csv.copy()
    csv['duration'] = csv['offset'] - csv['onset']
    csv = csv[csv['duration'] >= 0.4]
    csv['label'] = csv['label'].apply(lambda x: 'NSW' if x.startswith('NSW_') else x)
    
    print(f"After filtering and merging: {len(csv):,} samples")
    
    # Load the splits
    train_path = os.path.join(output_dir, 'train.csv')
    valid_path = os.path.join(output_dir, 'valid.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    if all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)
        print("Loaded existing train/valid/test splits")
    else:
        print("Splits not found. Please run the original script first.")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Original dataset distribution
    plot_class_distribution(
        original_csv, 
        'Original Dataset', 
        os.path.join(plots_dir, 'original_distribution.png')
    )
    
    # 2. Processed dataset distribution
    plot_class_distribution(
        csv, 
        'Processed Dataset (After Filtering & Merging)', 
        os.path.join(plots_dir, 'processed_distribution.png')
    )
    
    # 3. Dataset comparison across stages
    plot_dataset_comparison(
        original_csv, csv, train, valid, test,
        os.path.join(plots_dir, 'dataset_comparison.png')
    )
    
    # 4. Detailed imbalance analysis for final dataset
    plot_imbalance_metrics(
        csv, 
        'Final Processed Dataset',
        os.path.join(plots_dir, 'imbalance_analysis.png')
    )
    
    # 5. Individual split analyses
    for split_name, split_data in [('Train', train), ('Validation', valid), ('Test', test)]:
        plot_imbalance_metrics(
            split_data,
            f'{split_name} Split',
            os.path.join(plots_dir, f'{split_name.lower()}_imbalance_analysis.png')
        )
    
    print(f"\nAll plots saved to: {plots_dir}")
    print("\nSummary of class distribution:")
    print("=" * 50)
    print("Original dataset:")
    print(original_csv['label'].value_counts())
    print("\nProcessed dataset:")
    print(csv['label'].value_counts())
    print("\nTrain split:")
    print(train['label'].value_counts())
    print("\nValidation split:")
    print(valid['label'].value_counts())
    print("\nTest split:")
    print(test['label'].value_counts())

if __name__ == "__main__":
    main() 