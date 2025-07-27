#!/usr/bin/env python3
"""
Create a validation dataset from filtered extracted segments.
Processes WAV files from subfolders and creates a HuggingFace dataset for validation.
"""
import os
import random
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Audio, load_dataset, concatenate_datasets
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

def list_audio_files_recursive(directory: Union[str, Path], extensions=(".wav",)) -> List[str]:
    """
    Recursively list all audio files in directory and its subdirectories.
    
    Args:
        directory: Root directory to search
        extensions: Tuple of file extensions to include
        
    Returns:
        List of absolute paths to audio files
    """
    directory = Path(directory)
    audio_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(str(Path(root) / file))
    
    return audio_files

def process_in_batches(audio_paths, feature_extractor, batch_size=100, min_len=2.0, max_len=20.0, use_gpu=True):
    """
    Process audio files in batches to create dataset features.
    
    Args:
        audio_paths: List of audio file paths
        feature_extractor: Wav2Vec2 feature extractor
        batch_size: Number of files to process in each batch
        min_len: Minimum audio length in seconds
        max_len: Maximum audio length in seconds
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Concatenated dataset
    """
    def gen():
        for i in range(0, len(audio_paths), batch_size):
            yield audio_paths[i:i + batch_size]

    datasets_list = []
    features = Features({"audio": Audio(sampling_rate=feature_extractor.sampling_rate)})

    # Check GPU availability
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Processing {len(audio_paths)} audio files in batches of {batch_size}")
    
    for batch_paths in tqdm(gen(), total=(len(audio_paths) + batch_size - 1) // batch_size):
        ds = Dataset.from_dict({"audio": batch_paths}, features=features)

        def prepare(batch):
            sample = batch["audio"]
            
            # Convert to tensor and move to GPU if available
            if device.type == 'cuda':
                # Convert numpy array to tensor and move to GPU
                audio_array = torch.tensor(sample["array"], dtype=torch.float32, device=device)
                
                # Process on GPU
                inputs = feature_extractor(
                    audio_array.cpu().numpy(),  # Feature extractor expects numpy
                    sampling_rate=sample["sampling_rate"],
                    max_length=int(max_len * feature_extractor.sampling_rate),
                    truncation=True,
                    return_tensors="pt"  # Return PyTorch tensors
                )
                
                # Move input_values to GPU for processing
                input_values = inputs.input_values.to(device)
                
                return {
                    "input_values": input_values.cpu().numpy().astype(np.float32),
                    "input_length": input_values.shape[1]
                }
            else:
                # CPU processing (original code)
                inputs = feature_extractor(
                    sample["array"],
                    sampling_rate=sample["sampling_rate"],
                    max_length=int(max_len * feature_extractor.sampling_rate),
                    truncation=True
                )
                return {
                    "input_values": np.array(inputs.input_values[0], dtype=np.float32),
                    "input_length": len(inputs.input_values[0])
                }

        ds = ds.map(prepare, remove_columns=ds.column_names)

        ds = ds.filter(
            lambda x: x["input_length"] >= int(min_len * feature_extractor.sampling_rate)
        )
        ds = ds.remove_columns("input_length")
        datasets_list.append(ds)

    return concatenate_datasets(datasets_list)

def read_audio_dataset(directory: Union[str, Path], feature_extractor: Wav2Vec2FeatureExtractor, max_samples: int = None, use_gpu: bool = True):
    """
    Read audio files from directory and create dataset.
    
    Args:
        directory: Directory containing audio files (will search subdirectories)
        feature_extractor: Wav2Vec2 feature extractor
        max_samples: Maximum number of samples to include (random selection)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Processed dataset and list of selected file paths
    """
    print(f"Searching for audio files in: {directory}")
    audio_files = list_audio_files_recursive(directory)
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {directory}")
    
    # Randomly select samples if max_samples is specified
    if max_samples is not None:
        if len(audio_files) <= max_samples:
            print(f"Total files ({len(audio_files)}) <= max_samples ({max_samples}), using all files")
            selected_files = audio_files
        else:
            print(f"Randomly selecting {max_samples} files from {len(audio_files)} total files")
            selected_files = random.sample(audio_files, max_samples)
    else:
        selected_files = audio_files
    
    return process_in_batches(selected_files, feature_extractor, use_gpu=use_gpu), selected_files

def main():
    # Configuration
    dir_data = Path("/media/DOLPHIN1/new_wav2vec2_dataset/filtered_extracted_segments")
    output_dir = Path("/media/DOLPHIN1/new_wav2vec2_dataset/validation_dataset")
    max_samples = 2835  # Number of segments to randomly select
    use_gpu = True  # Set to False to use CPU only
    
    # Feature extractor configuration
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        str("/media/DOLPHIN/Analyses_alexis/source_code/preprocessor_config.json")
    )
    
    print("Creating validation dataset...")
    print(f"Input directory: {dir_data}")
    print(f"Output directory: {output_dir}")
    print(f"Target number of samples: {max_samples}")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Create validation dataset with random selection
    ds_validation, selected_files = read_audio_dataset(
        dir_data, feature_extractor, max_samples=max_samples, use_gpu=use_gpu
    )
    
    # Create dataset dictionary with only validation split
    ds_all = DatasetDict({"validation": ds_validation})
    
    # Save dataset
    print(f"Saving dataset to: {output_dir}")
    ds_all.save_to_disk(str(output_dir))
    
    # Save CSV with selected segments
    csv_path = output_dir / "selected_segments.csv"
    df_selected = pd.DataFrame({
        'file_path': selected_files,
        'filename': [Path(f).name for f in selected_files],
        'subfolder': [Path(f).parent.name for f in selected_files]
    })
    df_selected.to_csv(csv_path, index=False)
    print(f"Saved selected segments list to: {csv_path}")
    
    print(f"Dataset created successfully!")
    print(f"Validation set size: {len(ds_validation)} samples")
    print(f"Selected segments saved to: {csv_path}")

if __name__ == "__main__":
    main() 