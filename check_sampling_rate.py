import torchaudio
import pandas as pd
from pathlib import Path

def check_sampling_rate(csv_files):
    all_rates = set()
    
    for csv_file in csv_files:
        print(f"\nChecking files from {csv_file}")
        df = pd.read_csv(csv_file)
        
        for idx, row in df.iterrows():
            wav_path = row['path']
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                all_rates.add(sample_rate)
                if idx == 0:  # Print first file's rate
                    print(f"First file ({wav_path}): {sample_rate} Hz")
            except Exception as e:
                print(f"Error loading {wav_path}: {str(e)}")
    
    if all_rates:
        print(f"\nAll unique sampling rates found: {all_rates}")
        if len(all_rates) == 1:
            print("All files have the same sampling rate")
        else:
            print("Warning: Multiple sampling rates found!")

if __name__ == "__main__":
    csv_files = [
        "data/watkins/annotations.dolphins.train.csv",
        "data/watkins/annotations.dolphins.valid.csv",
        "data/watkins/annotations.dolphins.test.csv"
    ]
    check_sampling_rate(csv_files) 