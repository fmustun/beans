#!/usr/bin/env python3
"""
Remove this script after using it.

Script to split the binary dataset into train, validation, and test sets.
Labels: 0 for noise files, 1 for non-noise files.
"""

import os
import pandas as pd
import random
from pathlib import Path

def split_binary_dataset(data_dir, train_ratio=0.7, valid_ratio=0.20, test_ratio=0.10, random_seed=42):
    """
    Split the binary dataset into train, validation, and test sets.
    
    Args:
        data_dir (str): Path to the directory containing the audio files
        train_ratio (float): Proportion of data for training (default: 0.7)
        valid_ratio (float): Proportion of data for validation (default: 0.20)
        test_ratio (float): Proportion of data for testing (default: 0.10)
        random_seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_df, valid_df, test_df) - DataFrames with file paths and labels
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # Separate noise and non-noise files
    noise_files = [f for f in audio_files if 'noise' in f]
    non_noise_files = [f for f in audio_files if 'noise' not in f]
    
    print(f"Found {len(noise_files)} noise files and {len(non_noise_files)} non-noise files")
    
    # Shuffle both lists
    random.shuffle(noise_files)
    random.shuffle(non_noise_files)
    
    # Calculate split indices for each class
    noise_train_end = int(len(noise_files) * train_ratio)
    noise_valid_end = noise_train_end + int(len(noise_files) * valid_ratio)
    
    non_noise_train_end = int(len(non_noise_files) * train_ratio)
    non_noise_valid_end = non_noise_train_end + int(len(non_noise_files) * valid_ratio)
    
    # Split noise files
    noise_train = noise_files[:noise_train_end]
    noise_valid = noise_files[noise_train_end:noise_valid_end]
    noise_test = noise_files[noise_valid_end:]
    
    # Split non-noise files
    non_noise_train = non_noise_files[:non_noise_train_end]
    non_noise_valid = non_noise_files[non_noise_train_end:non_noise_valid_end]
    non_noise_test = non_noise_files[non_noise_valid_end:]
    
    # Create DataFrames
    def create_dataframe(file_list, label, data_dir):
        """Create a DataFrame with file paths and labels."""
        data = []
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            data.append({'file_path': file_path, 'label': label})
        return pd.DataFrame(data)
    
    # Create train DataFrame
    train_noise_df = create_dataframe(noise_train, 0, data_dir)
    train_non_noise_df = create_dataframe(non_noise_train, 1, data_dir)
    train_df = pd.concat([train_noise_df, train_non_noise_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Create validation DataFrame
    valid_noise_df = create_dataframe(noise_valid, 0, data_dir)
    valid_non_noise_df = create_dataframe(non_noise_valid, 1, data_dir)
    valid_df = pd.concat([valid_noise_df, valid_non_noise_df], ignore_index=True)
    valid_df = valid_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Create test DataFrame
    test_noise_df = create_dataframe(noise_test, 0, data_dir)
    test_non_noise_df = create_dataframe(non_noise_test, 1, data_dir)
    test_df = pd.concat([test_noise_df, test_non_noise_df], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return train_df, valid_df, test_df

def save_splits(train_df, valid_df, test_df, output_dir="data/binary"):
    """
    Save the train, validation, and test splits to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training data
        valid_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_dir (str): Directory to save the CSV files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    train_df.to_csv(os.path.join(output_dir, "annotations.train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "annotations.valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "annotations.test.csv"), index=False)
    
    print(f"Saved splits to {output_dir}:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(valid_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Print class distribution
    print("\nClass distribution:")
    print(f"  Train - Label 0: {len(train_df[train_df['label'] == 0])}, Label 1: {len(train_df[train_df['label'] == 1])}")
    print(f"  Valid - Label 0: {len(valid_df[valid_df['label'] == 0])}, Label 1: {len(valid_df[valid_df['label'] == 1])}")
    print(f"  Test  - Label 0: {len(test_df[test_df['label'] == 0])}, Label 1: {len(test_df[test_df['label'] == 1])}")

def main():
    """Main function to run the dataset splitting."""
    
    # Configuration
    data_dir = "data/binary"
    output_dir = "data/binary"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return
    
    print(f"Splitting dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Split the dataset
    train_df, valid_df, test_df = split_binary_dataset(data_dir)
    
    # Save the splits
    save_splits(train_df, valid_df, test_df, output_dir)
    
    print("\nDataset splitting completed successfully!")

if __name__ == "__main__":
    main() 