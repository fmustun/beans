import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
input_csv = 'data/dolphin_reef/all/all.csv'
output_dir = 'data/dolphin_reef/unbalanced_SW_NSW/'
os.makedirs(output_dir, exist_ok=True)

# Read data
csv = pd.read_csv(input_csv)

# Print original class distribution
print('Original class distribution:')
print(csv['label'].value_counts())

# Merge NSW_* labels into 'NSW'
csv['label'] = csv['label'].apply(lambda x: 'NSW' if x.startswith('NSW_') else x)

# Print class distribution after merging
print('\nClass distribution after merging NSW_* into NSW:')
print(csv['label'].value_counts())

# Split into train/valid/test (stratified, no balancing)
train, temp = train_test_split(csv, test_size=0.3, stratify=csv['label'], random_state=42)
valid, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

# Print class distribution in each split
print(f'\nTrain split class distribution ({len(train)} samples):')
print(train['label'].value_counts())
print(f'\nValid split class distribution ({len(valid)} samples):')
print(valid['label'].value_counts())
print(f'\nTest split class distribution ({len(test)} samples):')
print(test['label'].value_counts())

# Save splits
train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
valid.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print(f"\nSaved train/valid/test splits to {output_dir}")
