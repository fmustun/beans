#!/usr/bin/env python3
"""
Whistle Detection Script using Dolph2Vec model
Processes audio files in 0.3-second windows and detects whistles (class 1)
"""

import argparse
import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

from beans.models import Dolph2VecClassifier
from beans.datasets import ClassificationDataset


def load_model(model_path):
    """
    Load the trained Dolph2Vec model from checkpoint.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        
    Returns:
        tuple: (model, model_config, device) - Loaded model, its configuration, and device
    """
    print(f"Loading model from: {model_path}")
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    model_config = {
        'model_type': checkpoint.get('model_type', 'dolph2vec'),
        'dataset': checkpoint.get('dataset', 'unknown'),
        'task': checkpoint.get('task', 'classification'),
        'num_labels': checkpoint.get('num_labels', 2),
        'sample_rate': checkpoint.get('sample_rate', 44100),
        'valid_metric': checkpoint.get('valid_metric', 0.0),
        'test_metric': checkpoint.get('test_metric', 0.0)
    }
    
    print(f"Model configuration: {model_config}")
    
    # Create model
    model = Dolph2VecClassifier(
        sample_rate=model_config['sample_rate'],
        num_classes=model_config['num_labels']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move model to GPU if available
    model.eval()
    
    return model, model_config, device


def process_audio_file(audio_path, model, model_config, device, window_duration=0.4, overlap=0.1, confidence_threshold=0.6, batch_size=32):
    """
    Process an audio file in windows and detect whistles.
    Now supports batch processing for faster inference.
    """
    print(f"Processing audio file: {audio_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != model_config['sample_rate']:
        resampler = torchaudio.transforms.Resample(sample_rate, model_config['sample_rate'])
        waveform = resampler(waveform)
        sample_rate = model_config['sample_rate']
    
    # Move waveform to device
    waveform = waveform.to(device)
    
    # Calculate window parameters
    window_samples = int(window_duration * sample_rate)
    hop_samples = int(window_samples * (1 - overlap))
    
    # Prepare all windows and timestamps
    windows = []
    timestamps = []
    window_starts = []
    window_ends = []
    num_windows = (waveform.shape[1] - window_samples) // hop_samples + 1
    print(f"Processing {num_windows} windows of {window_duration}s each (batch size: {batch_size})")
    for i in range(num_windows):
        start_sample = i * hop_samples
        end_sample = start_sample + window_samples
        if end_sample > waveform.shape[1]:
            break
        window = waveform[:, start_sample:end_sample].squeeze(0)
        windows.append(window)
        timestamps.append(start_sample / sample_rate)
        window_starts.append(start_sample)
        window_ends.append(end_sample)
    if not windows:
        print("No valid windows extracted from audio.")
        return []
    windows_tensor = torch.stack(windows)  # [num_windows, window_samples]
    results = []
    with torch.no_grad():
        for batch_start in range(0, len(windows_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(windows_tensor))
            window_batch = windows_tensor[batch_start:batch_end].to(device)
            # Prepare dummy targets
            if model_config['task'] == 'classification':
                dummy_target = torch.zeros(window_batch.size(0), dtype=torch.long, device=device)
                _, logits = model(window_batch, dummy_target)
                probabilities = torch.softmax(logits, dim=1)
                confidences = probabilities[:, 1].detach().cpu().numpy()
                predictions = (confidences > confidence_threshold).astype(int)
            else:
                dummy_target = torch.zeros(window_batch.size(0), model_config['num_labels'], device=device)
                _, logits = model(window_batch, dummy_target)
                probabilities = torch.sigmoid(logits)
                confidences = probabilities[:, 1].detach().cpu().numpy()
                predictions = (confidences > confidence_threshold).astype(int)
            for idx, (conf, pred) in enumerate(zip(confidences, predictions)):
                global_idx = batch_start + idx
                results.append({
                    'timestamp': timestamps[global_idx],
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'window_start': window_starts[global_idx],
                    'window_end': window_ends[global_idx]
                })
    return results


def save_results(results, output_path, window_duration=0.4):
    """
    Save detection results to CSV file.
    
    Args:
        results (list): List of detection results
        output_path (str): Path to save the CSV file
        window_duration (float): Duration of each window in seconds (default: 0.4)
    """
    if not results:
        print("No results to save - all windows failed processing")
        # Create empty CSV with correct columns
        empty_df = pd.DataFrame(columns=['start_time', 'end_time', 'confidence'])
        empty_df.to_csv(output_path, index=False)
        return
    
    df = pd.DataFrame(results)
    
    # Filter for whistle detections (class 1)
    whistle_detections = df[df['prediction'] == 1].copy()
    
    if len(whistle_detections) == 0:
        print("No whistle detections found")
        # Create empty CSV with correct columns
        empty_df = pd.DataFrame(columns=['start_time', 'end_time', 'confidence'])
        empty_df.to_csv(output_path, index=False)
        return
    
    # Add window end timestamp using the actual window_duration
    whistle_detections['window_end_time'] = whistle_detections['timestamp'] + window_duration
    
    # Select relevant columns for output
    output_df = whistle_detections[['timestamp', 'window_end_time', 'confidence']].copy()
    output_df.columns = ['start_time', 'end_time', 'confidence']
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(output_df)} whistle detections to: {output_path}")
    print(f"Total windows processed: {len(results)}")
    print(f"Whistle detection rate: {len(output_df)/len(results)*100:.2f}%")


def process_folder(input_folder, output_folder, model, model_config, device, window_duration=0.4, overlap=0.1, confidence_threshold=0.6, batch_size=32):
    """
    Process all WAV files in a folder and generate prediction CSVs.
    
    Args:
        input_folder (str): Path to the folder containing WAV files
        output_folder (str): Path to the folder where CSV files will be saved
        model: Loaded Dolph2Vec model
        model_config (dict): Model configuration
        device: Device to run inference on (cuda/cpu)
        window_duration (float): Duration of each window in seconds
        overlap (float): Overlap between windows (0.0 = no overlap)
        confidence_threshold (float): Confidence threshold for whistle detection
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all WAV files in the folder
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in folder: {input_folder}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Process each WAV file
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        
        # Create output CSV filename with "_predictions" suffix in the output folder
        output_csv = os.path.join(output_folder, f"{base_name}_predictions.csv")
        
        # Skip if output already exists
        if os.path.exists(output_csv):
            print(f"Output already exists for {wav_file}, skipping.")
            continue
        
        print(f"\nProcessing: {wav_file}")
        print(f"Output will be saved to: {output_csv}")
        
        try:
            # Process the audio file
            results = process_audio_file(
                audio_path=wav_file,
                model=model,
                model_config=model_config,
                device=device,
                window_duration=window_duration,
                overlap=overlap,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"Error processing {wav_file}: {e}. Skipping this file.")
            continue
        
        # Save results
        save_results(results, output_csv, window_duration)


def main():
    parser = argparse.ArgumentParser(description='Whistle Detection using Dolph2Vec')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--input-folder', type=str, required=True,
                       help='Path to the folder containing WAV files to process')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Path to the folder where CSV prediction files will be saved')
    parser.add_argument('--window-duration', type=float, default=0.4,
                       help='Duration of each window in seconds (default: 0.4)')
    parser.add_argument('--overlap', type=float, default=0.1,
                       help='Overlap between windows (0.0 = no overlap, default: 0.1)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Confidence threshold for whistle detection (default: 0.6)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for model inference (default: 32)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder not found: {args.input_folder}")
        return
    
    # Check if output folder can be created (will be created if it doesn't exist)
    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output folder {args.output_folder}: {e}")
        return
    
    # Load model
    model, model_config, device = load_model(args.model)
    
    # Process all WAV files in the folder
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model=model,
        model_config=model_config,
        device=device,
        window_duration=args.window_duration,
        overlap=args.overlap,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 