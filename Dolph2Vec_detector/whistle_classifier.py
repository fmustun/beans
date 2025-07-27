#!/usr/bin/env python3
"""
Whistle Classification Script using Dolph2Vec model
Classifies whistle segments (from a CSV) into 10 classes using a trained Dolph2VecClassifier
"""

import argparse
import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from pathlib import Path

from beans.models import Dolph2VecClassifier


def load_model(model_path):
    print(f"Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = {
        'model_type': checkpoint.get('model_type', 'dolph2vec'),
        'dataset': checkpoint.get('dataset', 'unknown'),
        'task': checkpoint.get('task', 'classification'),
        'num_labels': checkpoint.get('num_labels', 8),
        'sample_rate': checkpoint.get('sample_rate', 44100),
        'valid_metric': checkpoint.get('valid_metric', 0.0),
        'test_metric': checkpoint.get('test_metric', 0.0)
    }
    print(f"Model configuration: {model_config}")
    model = Dolph2VecClassifier(
        sample_rate=model_config['sample_rate'],
        num_classes=model_config['num_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_config, device


def classify_segment(waveform, sample_rate, model, model_config, device):
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if necessary
    if sample_rate != model_config['sample_rate']:
        resampler = torchaudio.transforms.Resample(sample_rate, model_config['sample_rate'])
        waveform = resampler(waveform)
        sample_rate = model_config['sample_rate']
    waveform = waveform.to(device)
    waveform = waveform.squeeze(0)
    try:
        with torch.no_grad():
            dummy_target = torch.zeros(1, dtype=torch.long, device=device)
            waveform_batch = waveform.unsqueeze(0)
            _, logits = model(waveform_batch, dummy_target)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
            all_probabilities = probabilities[0].cpu().numpy().tolist()
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'segment_duration': waveform.shape[0] / sample_rate
            }
    except Exception as e:
        print(f"Error classifying segment: {e}")
        return {
            'predicted_class': -1,
            'confidence': 0.0,
            'all_probabilities': [0.0] * model_config['num_labels'],
            'segment_duration': 0.0
        }
    return result


def process_csv(csv_path, recordings_folder, model, model_config, device, output_path):
    df = pd.read_csv(csv_path)
    results = []
    for idx, row in df.iterrows():
        recording_name = str(row['recording_name'])
        onset = float(row['onset'])
        offset = float(row['offset'])
        audio_path = os.path.join(recordings_folder, recording_name + '.wav')
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            result = {
                'predicted_class': -1,
                'confidence': 0.0,
                'segment_duration': 0.0,
            }
            for i in range(model_config['num_labels']):
                result[f'class_{i}_probability'] = 0.0
            results.append({**row, **result})
            continue
        try:
            info = torchaudio.info(audio_path)
            sample_rate = info.sample_rate
            start_frame = int(onset * sample_rate)
            num_frames = int((offset - onset) * sample_rate)
            waveform, _ = torchaudio.load(audio_path, frame_offset=start_frame, num_frames=num_frames)
            result = classify_segment(waveform, sample_rate, model, model_config, device)
        except Exception as e:
            print(f"Error processing {audio_path} ({onset}-{offset}): {e}")
            result = {
                'predicted_class': -1,
                'confidence': 0.0,
                'segment_duration': 0.0,
            }
            for i in range(model_config['num_labels']):
                result[f'class_{i}_probability'] = 0.0
            results.append({**row, **result})
            continue
        # Add class probabilities as separate columns
        for i, prob in enumerate(result['all_probabilities']):
            result[f'class_{i}_probability'] = prob
        del result['all_probabilities']
        results.append({**row, **result})
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Saved classification results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Whistle Classification using Dolph2Vec on segments from CSV')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file with whistle segments')
    parser.add_argument('--recordings-folder', type=str, required=True, help='Path to the folder containing audio recordings')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    if not os.path.exists(args.recordings_folder):
        print(f"Error: Recordings folder not found: {args.recordings_folder}")
        return
    model, model_config, device = load_model(args.model)
    process_csv(args.csv, args.recordings_folder, model, model_config, device, args.output)


if __name__ == "__main__":
    main() 