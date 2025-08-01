#!/usr/bin/env python3
"""
Find recordings with no whistle detections in a folder tree.
For each subfolder, runs the whistle detector on all .wav files and outputs a CSV listing files with no detections.
Also saves individual CSV results for each recording.
"""
import os
import argparse
import pandas as pd
from whistle_detector import load_model, process_audio_file, save_results

def find_no_whistle_files(root_dir, model_path, window_duration=0.4, overlap=0.1, confidence_threshold=0.6):
    # Load model once
    model, model_config, device = load_model(model_path)
    
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_dir):
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        if not wav_files:
            continue
        no_detection_files = []
        print(f"Processing folder: {dirpath} ({len(wav_files)} wav files)")
        for wav_file in wav_files:
            wav_path = os.path.join(dirpath, wav_file)
            try:
                results = process_audio_file(
                    audio_path=wav_path,
                    model=model,
                    model_config=model_config,
                    device=device,
                    window_duration=window_duration,
                    overlap=overlap,
                    confidence_threshold=confidence_threshold
                )
                
                # Save individual CSV results for this recording
                csv_filename = os.path.splitext(wav_file)[0] + '_whistle_detections.csv'
                csv_path = os.path.join(dirpath, csv_filename)
                save_results(results, csv_path, window_duration)
                
                # If no window has prediction==1, add to list
                if not any(r['prediction'] == 1 for r in results):
                    no_detection_files.append(wav_file)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
        # Write CSV if any files with no detections
        if no_detection_files:
            csv_path = os.path.join(dirpath, 'no_whistle_detections.csv')
            pd.DataFrame({'file': no_detection_files}).to_csv(csv_path, index=False)
            print(f"Wrote {len(no_detection_files)} files with no detections to {csv_path}")
        else:
            print(f"All files in {dirpath} had whistle detections.")

def main():
    parser = argparse.ArgumentParser(description='Find recordings with no whistle detections in a folder tree.')
    parser.add_argument('--root', type=str, required=True, help='Root directory to search (contains subfolders with wav files)')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--window-duration', type=float, default=0.4, help='Window duration in seconds (default: 0.4)')
    parser.add_argument('--overlap', type=float, default=0.1, help='Window overlap (default: 0.1)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Confidence threshold (default: 0.6)')
    args = parser.parse_args()
    
    find_no_whistle_files(
        root_dir=args.root,
        model_path=args.model,
        window_duration=args.window_duration,
        overlap=args.overlap,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main() 