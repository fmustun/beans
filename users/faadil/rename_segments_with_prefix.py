#!/usr/bin/env python3
"""
Rename WAV segments by adding the subfolder name as a prefix.
Walks through all subfolders in filtered_extracted_segments and renames WAV files
to include the parent folder name as a prefix.
"""
import os
import argparse
import shutil
from pathlib import Path

def rename_segments_with_prefix(root_dir):
    """
    Rename WAV files by adding the subfolder name as a prefix.
    
    Args:
        root_dir (str): Root directory containing filtered_extracted_segments
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory not found: {root_path}")
        return
    
    print(f"Processing segments in: {root_path}")
    
    total_files_renamed = 0
    total_files_processed = 0
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirpath = Path(dirpath)
        
        # Skip the root directory itself
        if dirpath == root_path:
            continue
            
        # Find WAV files
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        
        if not wav_files:
            continue
            
        # Get the subfolder name (the immediate parent folder)
        subfolder_name = dirpath.name
        
        print(f"\nProcessing folder: {dirpath} ({len(wav_files)} WAV files)")
        print(f"Subfolder name: {subfolder_name}")
        
        for wav_file in wav_files:
            total_files_processed += 1
            
            wav_path = dirpath / wav_file
            
            # Create new filename with subfolder prefix
            new_wav_name = f"{subfolder_name}_{wav_file}"
            new_wav_path = dirpath / new_wav_name
            
            # Check if the new filename already exists
            if new_wav_path.exists():
                print(f"  Warning: Target file already exists, skipping: {new_wav_name}")
                continue
            
            try:
                # Rename the file
                wav_path.rename(new_wav_path)
                total_files_renamed += 1
                print(f"  ✓ Renamed: {wav_file} -> {new_wav_name}")
                
                # Also rename the corresponding CSV file if it exists
                csv_file = wav_file.replace('.wav', '_whistle_detections.csv')
                csv_path = dirpath / csv_file
                
                if csv_path.exists():
                    new_csv_name = f"{subfolder_name}_{csv_file}"
                    new_csv_path = dirpath / new_csv_name
                    
                    if not new_csv_path.exists():
                        csv_path.rename(new_csv_path)
                        print(f"    ✓ Renamed CSV: {csv_file} -> {new_csv_name}")
                    else:
                        print(f"    Warning: Target CSV file already exists, skipping: {new_csv_name}")
                
            except Exception as e:
                print(f"  Error renaming {wav_file}: {e}")
                continue
    
    print(f"\nSummary:")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files renamed: {total_files_renamed}")
    print(f"Rename rate: {total_files_renamed/total_files_processed*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Rename WAV segments by adding subfolder name as prefix')
    parser.add_argument('--root', type=str, 
                       default='/media/DOLPHIN1/new_wav2vec2_dataset/filtered_extracted_segments',
                       help='Root directory containing filtered_extracted_segments')
    
    args = parser.parse_args()
    
    rename_segments_with_prefix(args.root)

if __name__ == "__main__":
    main() 