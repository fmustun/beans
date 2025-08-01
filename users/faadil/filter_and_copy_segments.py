#!/usr/bin/env python3
"""
Filter and copy segments based on whistle detection confidence.
Walks through all folders in extracted_segments, filters CSV files by confidence threshold,
and copies corresponding WAV files if they have enough high-confidence detections.
"""
import os
import argparse
import pandas as pd
import shutil
from pathlib import Path

def filter_and_copy_segments(source_root, target_root, confidence_threshold=0.8, min_windows=5):
    """
    Filter CSV files by confidence threshold and copy corresponding WAV files.
    
    Args:
        source_root (str): Root directory containing extracted_segments
        target_root (str): Root directory for filtered_extracted_segments
        confidence_threshold (float): Minimum confidence threshold (default: 0.8)
        min_windows (int): Minimum number of windows required after filtering (default: 5)
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing segments from: {source_path}")
    print(f"Copying filtered segments to: {target_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Minimum windows required: {min_windows}")
    
    total_files_processed = 0
    total_files_copied = 0
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(source_path):
        dirpath = Path(dirpath)
        
        # Find CSV files that correspond to WAV files
        csv_files = [f for f in filenames if f.endswith('_whistle_detections.csv')]
        
        if not csv_files:
            continue
            
        print(f"\nProcessing folder: {dirpath} ({len(csv_files)} CSV files)")
        
        # Create corresponding target directory
        relative_path = dirpath.relative_to(source_path)
        target_dir = target_path / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for csv_file in csv_files:
            total_files_processed += 1
            
            # Extract base name (remove '_whistle_detections.csv')
            base_name = csv_file.replace('_whistle_detections.csv', '')
            wav_file = f"{base_name}.wav"
            csv_path = dirpath / csv_file
            wav_path = dirpath / wav_file
            
            # Check if corresponding WAV file exists
            if not wav_path.exists():
                print(f"  Warning: WAV file not found for {csv_file}: {wav_file}")
                continue
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    print(f"  Skipping {csv_file}: Empty CSV file")
                    continue
                
                # Filter by confidence threshold
                filtered_df = df[df['confidence'] >= confidence_threshold]
                
                print(f"  {csv_file}: {len(df)} total windows, {len(filtered_df)} windows >= {confidence_threshold}")
                
                # Check if enough windows remain
                if len(filtered_df) >= min_windows:
                    # Copy WAV file to target directory
                    target_wav_path = target_dir / wav_file
                    shutil.copy2(wav_path, target_wav_path)
                    
                    # Also copy the filtered CSV file
                    target_csv_path = target_dir / csv_file
                    filtered_df.to_csv(target_csv_path, index=False)
                    
                    total_files_copied += 1
                    print(f"    ✓ Copied {wav_file} ({len(filtered_df)} windows)")
                else:
                    print(f"    ✗ Skipped {wav_file} (only {len(filtered_df)} windows >= {confidence_threshold})")
                    
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")
                continue
    
    print(f"\nSummary:")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files copied: {total_files_copied}")
    print(f"Copy rate: {total_files_copied/total_files_processed*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Filter and copy segments based on whistle detection confidence')
    parser.add_argument('--source', type=str, 
                       default='/media/DOLPHIN1/new_wav2vec2_dataset/extracted_segments',
                       help='Source directory containing extracted_segments')
    parser.add_argument('--target', type=str,
                       default='/media/DOLPHIN1/new_wav2vec2_dataset/filtered_extracted_segments',
                       help='Target directory for filtered_extracted_segments')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                       help='Minimum confidence threshold (default: 0.8)')
    parser.add_argument('--min-windows', type=int, default=5,
                       help='Minimum number of windows required after filtering (default: 5)')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory not found: {args.source}")
        return
    
    filter_and_copy_segments(
        source_root=args.source,
        target_root=args.target,
        confidence_threshold=args.confidence_threshold,
        min_windows=args.min_windows
    )

if __name__ == "__main__":
    main() 