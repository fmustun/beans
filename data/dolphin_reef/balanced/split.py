import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def split_csv(file_path, train_size=0.7, val_size=0.1, test_size=0.2, seed=42, label_column="label"):
    """
    Split a CSV file into train, validation, and test sets with balanced class distribution.

    Parameters:
    - file_path: Path to the input CSV file
    - train_size: Proportion for training set (default: 0.7)
    - val_size: Proportion for validation set (default: 0.1)
    - test_size: Proportion for test set (default: 0.2)
    - seed: Random seed for reproducibility
    - label_column: Column name containing class labels

    Returns:
    - train, val, test DataFrames
    """
    # Check that splits add up to 1.0
    total = train_size + val_size + test_size
    assert abs(total - 1.0) < 1e-6, "Train, validation, and test sizes must add up to 1.0!"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # First split: separate train from (val+test)
    val_test_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_size,
        random_state=seed,
        stratify=df[label_column]
    )

    # Second split: separate val and test from temp
    val_ratio = val_size / val_test_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1-val_ratio),
        random_state=seed,
        stratify=temp_df[label_column]
    )

    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Split CSV into balanced train, validation, and test sets.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of training set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of validation set")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output CSV files")
    parser.add_argument("--label_column", type=str, default="label", help="Column name containing class labels")

    args = parser.parse_args()

    train_df, val_df, test_df = split_csv(
        args.file_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        label_column=args.label_column
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to specified output directory
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    print(f"Split complete! Files saved to {args.output_dir}")
    print(f"Train set: {len(train_df)} samples ({train_df[args.label_column].value_counts().to_dict()})")
    print(f"Validation set: {len(val_df)} samples ({val_df[args.label_column].value_counts().to_dict()})")
    print(f"Test set: {len(test_df)} samples ({test_df[args.label_column].value_counts().to_dict()})")

if __name__ == "__main__":
    main()

