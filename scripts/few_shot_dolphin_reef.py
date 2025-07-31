#!/usr/bin/env python3
"""
Script for few-shot classification experiments on the dolphin_reef dataset.
Modifies the dataset in memory for efficiency.
"""

import argparse
import os
import pandas as pd
import numpy as np
import random
import subprocess
import json
import copy
import sys
import yaml
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# Import functions from evaluate.py
sys.path.append('.')
from scripts.evaluate import (
    set_all_seeds, 
    read_datasets, 
    eval_pytorch_model, 
    train_pytorch_model
)
from beans.datasets import ClassificationDataset, RecognitionDataset
from beans.metrics import Accuracy, MeanAveragePrecision
from beans.models import (
    BiolingualClassifier, 
    AvesClassifier, 
    Dolph2VecClassifier
)

class FewShotClassificationDataset(ClassificationDataset):
    """
    Modified ClassificationDataset that supports few-shot sampling.
    """
    def __init__(self, base_dataset, samples_per_class, seed=42):
        """
        Create a few-shot subset of the base dataset.
        
        Args:
            base_dataset: Original ClassificationDataset
            samples_per_class: Number of samples to keep per class
            seed: Random seed for reproducibility
        """
        # Copy the base dataset attributes
        self.sample_rate = base_dataset.sample_rate
        self.max_duration = base_dataset.max_duration
        self.feature_type = base_dataset.feature_type
        
        # Get all unique labels and their indices
        unique_labels = list(set(base_dataset.ys))
        label_indices = {label: [] for label in unique_labels}
        
        for i, label in enumerate(base_dataset.ys):
            label_indices[label].append(i)
        
        # Sample indices for each class
        selected_indices = []
        random.seed(seed)
        np.random.seed(seed)
        
        for label in unique_labels:
            indices = label_indices[label]
            if len(indices) <= samples_per_class:
                # If we have fewer samples than requested, use all available
                selected_indices.extend(indices)
            else:
                # Randomly sample the requested number
                sampled_indices = random.sample(indices, samples_per_class)
                selected_indices.extend(sampled_indices)
        
        # Shuffle the selected indices
        random.shuffle(selected_indices)
        
        # Create the subset
        self.xs = [base_dataset.xs[i] for i in selected_indices]
        self.ys = [base_dataset.ys[i] for i in selected_indices]
        
        print(f"Created few-shot dataset with {len(self.xs)} samples ({samples_per_class} per class)")
        
        # Print class distribution
        from collections import Counter
        class_counts = Counter(self.ys)
        print("Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} samples")


def run_few_shot_experiment(samples_per_class, model_type, classifier_type='linear',
                          freeze_feature_encoder=False, epochs=50, batch_size=32,
                          output_dir="few_shot_results", seed=42, task='classification',
                          dolph2vec_variant='base', lrs="[0.0001, 0.00005, 0.00001]"):
    """
    Run a few-shot classification experiment.
    
    Args:
        samples_per_class: Number of samples per class
        model_type: Type of model to use
        classifier_type: Type of classifier head
        freeze_feature_encoder: Whether to freeze feature encoder
        epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Output directory for results
        seed: Random seed
        task: Task type ('classification' or 'detection')
    """
    # Set random seed
    set_all_seeds(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"log_{model_type}_{samples_per_class}samples_{timestamp}.jsonl")
    log_file = open(log_file_path, 'w')
    
    # Log experiment configuration
    config = {
        'experiment_config': {
            'samples_per_class': samples_per_class,
            'model_type': model_type,
            'classifier_type': classifier_type,
            'freeze_feature_encoder': freeze_feature_encoder,
            'epochs': epochs,
            'batch_size': batch_size,
            'seed': seed,
            'task': task,
            'dolph2vec_variant': dolph2vec_variant if model_type == 'dolph2vec' else None,
            'lrs': lrs
        }
    }
    log_file.write(json.dumps(config) + '\n')
    
    print(f"Running few-shot experiment: {model_type} with {samples_per_class} samples per class")
    print(f"Log file: {log_file_path}")
    
    # Read dataset configuration from datasets.yml
    datasets = read_datasets('datasets.yml')
    dataset = datasets['dolphin_reef']
    num_labels = dataset['num_labels']
    
    # Determine feature type based on model
    if model_type in ['biolingual', 'aves', 'dolph2vec']:
        feature_type = 'waveform'
    else:
        feature_type = 'mfcc'
    
    sample_rate = dataset['sample_rate'] if model_type != 'biolingual' else 48000
    
    # Load original datasets
    print("Loading original datasets...")
    if dataset['type'] == 'classification':
        dataset_train_full = ClassificationDataset(
            metadata_path=dataset['train_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=sample_rate,
            max_duration=dataset['max_duration'],
            feature_type=feature_type
        )
        
        dataset_valid = ClassificationDataset(
            metadata_path=dataset['valid_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=sample_rate,
            max_duration=dataset['max_duration'],
            feature_type=feature_type
        )
        
        dataset_test = ClassificationDataset(
            metadata_path=dataset['test_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=sample_rate,
            max_duration=dataset['max_duration'],
            feature_type=feature_type
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset['type']}")
    
    # Create few-shot training dataset
    print(f"Creating few-shot dataset with {samples_per_class} samples per class...")
    dataset_train = FewShotClassificationDataset(dataset_train_full, samples_per_class, seed)
    
    # Create data loaders
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=generator
    )
    
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Set up metric
    if task == 'classification':
        Metric = Accuracy
    elif task == 'detection':
        Metric = MeanAveragePrecision
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model using the imported function
    # Create a mock args object for PyTorch models
    class MockArgs:
        def __init__(self):
            self.lrs = lrs
            self.model_type = model_type
            self.task = task
            self.classifier_type = classifier_type
            self.freeze_feature_encoder = freeze_feature_encoder
            self.train_feature_encoder = not freeze_feature_encoder
            self.dolph2vec_variant = dolph2vec_variant if model_type == 'dolph2vec' else None
            self.epochs = epochs
    
    mock_args = MockArgs()
    model, valid_metric_best = train_pytorch_model(
        args=mock_args,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        num_labels=num_labels,
        metric_factory=Metric,
        sample_rate=sample_rate,
        device=device,
        log_file=log_file,
        freeze_feature_encoder=freeze_feature_encoder
    )
    
    if dataloader_test is not None:
        _, test_metric = eval_pytorch_model(
            model=model,
            dataloader=dataloader_test,
            metric_factory=Metric,
            device=device,
            desc='test'
        )
    
    # Log final results
    final_results = {
        'final_results': {
            'valid_metric_best': valid_metric_best,
            'test_metric': test_metric if 'test_metric' in locals() else None
        }
    }
    log_file.write(json.dumps(final_results) + '\n')
    
    print(f"Best validation metric: {valid_metric_best:.4f}")
    if 'test_metric' in locals():
        print(f"Test metric: {test_metric:.4f}")
    
    log_file.close()
    
    # Return results for summary
    results = {
        'samples_per_class': samples_per_class,
        'model_type': model_type,
        'classifier_type': classifier_type,
        'freeze_feature_encoder': freeze_feature_encoder,
        'valid_metric_best': valid_metric_best,
        'test_metric': test_metric if 'test_metric' in locals() else None,
        'log_file': log_file_path
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Few-shot classification experiments on dolphin_reef")
    parser.add_argument("--samples-per-class", type=int, nargs="+", 
                       default=[50, 100, 200, 300], 
                       help="Number of samples per class to test")
    parser.add_argument("--model-type", type=str, default="dolph2vec",
                       choices=["dolph2vec", "biolingual", "aves"],
                       help="Model type to use")
    parser.add_argument("--dolph2vec-variant", choices=['base', '32', '128', 'clean'], 
                       default='base', help='Dolph2Vec model variant (only used when model-type is dolph2vec)')
    parser.add_argument("--classifier-type", type=str, default="linear",
                       choices=["mlp", "linear"],
                       help="Type of classifier head")
    parser.add_argument("--freeze-feature-encoder", action="store_true",
                       help="Freeze the feature encoder")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lrs", type=str, default="[0.0001, 0.00005, 0.00001]",
                       help="Learning rates to try (as a string list)")
    parser.add_argument("--output-dir", type=str, default="few_shot_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--task", type=str, default="classification",
                       choices=["classification", "detection"],
                       help="Task type")
    
    args = parser.parse_args()
    
    # Store all results
    all_results = []
    
    # Run experiments for each number of samples per class
    for samples_per_class in args.samples_per_class:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {samples_per_class} samples per class")
        print(f"{'='*60}")
        
        try:
            results = run_few_shot_experiment(
                samples_per_class=samples_per_class,
                model_type=args.model_type,
                classifier_type=args.classifier_type,
                freeze_feature_encoder=args.freeze_feature_encoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                seed=args.seed,
                task=args.task,
                dolph2vec_variant=args.dolph2vec_variant,
                lrs=args.lrs
            )
            all_results.append(results)
        except Exception as e:
            print(f"Experiment failed: {e}")
            all_results.append({
                'samples_per_class': samples_per_class,
                'error': str(e)
            })
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        if 'error' in result:
            print(f"{result['samples_per_class']:3d} samples/class: FAILED - {result['error']}")
        else:
            print(f"{result['samples_per_class']:3d} samples/class: SUCCESS - Valid: {result['valid_metric_best']:.4f}, Test: {result['test_metric']:.4f}")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main() 