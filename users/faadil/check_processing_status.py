#!/usr/bin/env python3
"""
Check processing status and statistics for whistle detection.
Shows which folders have been processed and statistics on recordings with no whistles.
"""

import os
import pandas as pd
import argparse
from pathlib import Path

def check_processing_status(root_dir):
    """
    Check which folders have been processed and provide statistics.
    
    Args:
        root_dir (str): Root directory to check
        
    Returns:
        dict: Statistics about processing status
    """
    processed_folders = []
    unprocessed_folders = []
    total_no_whistle_files = 0
    folder_stats = []
    
    print(f"Checking processing status in: {root_dir}")
    print("=" * 60)
    
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue
            
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        csv_file = 'no_whistle_detections.csv'
        
        if wav_files:  # Only process folders with wav files
            csv_path = os.path.join(dirpath, csv_file)
            
            if os.path.exists(csv_path):
                # Folder has been processed
                processed_folders.append(dirpath)
                
                # Read the CSV to get statistics
                try:
                    df = pd.read_csv(csv_path)
                    no_whistle_count = len(df)
                    total_files = len(wav_files)
                    whistle_count = total_files - no_whistle_count
                    
                    folder_name = os.path.basename(dirpath)
                    folder_stats.append({
                        'folder': folder_name,
                        'total_files': total_files,
                        'no_whistle_files': no_whistle_count,
                        'whistle_files': whistle_count,
                        'no_whistle_percentage': (no_whistle_count / total_files) * 100
                    })
                    
                    total_no_whistle_files += no_whistle_count
                    
                    print(f"✓ {folder_name}: {no_whistle_count}/{total_files} files with no whistles ({no_whistle_count/total_files*100:.1f}%)")
                    
                except Exception as e:
                    print(f"✗ {os.path.basename(dirpath)}: Error reading CSV - {e}")
                    
            else:
                # Folder has not been processed
                unprocessed_folders.append(dirpath)
                folder_name = os.path.basename(dirpath)
                print(f"✗ {folder_name}: Not processed ({len(wav_files)} wav files)")
    
    # Create summary statistics
    total_processed_folders = len(processed_folders)
    total_unprocessed_folders = len(unprocessed_folders)
    total_folders = total_processed_folders + total_unprocessed_folders
    
    if folder_stats:
        df_stats = pd.DataFrame(folder_stats)
        total_files_processed = df_stats['total_files'].sum()
        overall_no_whistle_percentage = (total_no_whistle_files / total_files_processed) * 100
    else:
        total_files_processed = 0
        overall_no_whistle_percentage = 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total folders: {total_folders}")
    print(f"Processed folders: {total_processed_folders}")
    print(f"Unprocessed folders: {total_unprocessed_folders}")
    print(f"Processing progress: {total_processed_folders/total_folders*100:.1f}%")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files with no whistles: {total_no_whistle_files}")
    print(f"Overall no-whistle percentage: {overall_no_whistle_percentage:.1f}%")
    
    if folder_stats:
        print(f"\nAverage no-whistle percentage per folder: {df_stats['no_whistle_percentage'].mean():.1f}%")
        print(f"Median no-whistle percentage per folder: {df_stats['no_whistle_percentage'].median():.1f}%")
        print(f"Min no-whistle percentage: {df_stats['no_whistle_percentage'].min():.1f}%")
        print(f"Max no-whistle percentage: {df_stats['no_whistle_percentage'].max():.1f}%")
    
    # Return statistics
    stats = {
        'total_folders': total_folders,
        'processed_folders': total_processed_folders,
        'unprocessed_folders': total_unprocessed_folders,
        'processing_progress': total_processed_folders/total_folders*100,
        'total_files_processed': total_files_processed,
        'total_no_whistle_files': total_no_whistle_files,
        'overall_no_whistle_percentage': overall_no_whistle_percentage,
        'folder_stats': folder_stats,
        'unprocessed_folder_names': [os.path.basename(f) for f in unprocessed_folders]
    }
    
    return stats

def save_detailed_report(stats, output_path):
    """
    Save a detailed report to CSV.
    
    Args:
        stats (dict): Statistics from check_processing_status
        output_path (str): Path to save the detailed report
    """
    if stats['folder_stats']:
        df_detailed = pd.DataFrame(stats['folder_stats'])
        df_detailed = df_detailed.sort_values('no_whistle_percentage', ascending=False)
        df_detailed.to_csv(output_path, index=False)
        print(f"\nDetailed report saved to: {output_path}")
    else:
        print("\nNo processed folders found to create detailed report.")

def main():
    parser = argparse.ArgumentParser(description='Check processing status and statistics for whistle detection.')
    parser.add_argument('--root', type=str, required=True, 
                       help='Root directory to check (contains subfolders with wav files)')
    parser.add_argument('--output', type=str, default='processing_report.csv',
                       help='Path to save detailed report (default: processing_report.csv)')
    
    args = parser.parse_args()
    
    # Check processing status
    stats = check_processing_status(args.root)
    
    # Save detailed report
    save_detailed_report(stats, args.output)
    
    # Show unprocessed folders if any
    if stats['unprocessed_folder_names']:
        print(f"\nUnprocessed folders ({len(stats['unprocessed_folder_names'])}):")
        for folder in stats['unprocessed_folder_names'][:10]:  # Show first 10
            print(f"  - {folder}")
        if len(stats['unprocessed_folder_names']) > 10:
            print(f"  ... and {len(stats['unprocessed_folder_names']) - 10} more")

if __name__ == "__main__":
    main() 