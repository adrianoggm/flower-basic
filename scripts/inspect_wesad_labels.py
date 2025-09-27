#!/usr/bin/env python3
"""
WESAD Data Inspector - Check label structure
"""

import pickle
import numpy as np
from pathlib import Path

def inspect_wesad_labels():
    """Inspect WESAD label structure."""
    data_dir = Path("data/WESAD")
    
    for subject_id in ['S2', 'S3', 'S4']:
        subject_path = data_dir / subject_id / f"{subject_id}.pkl"
        
        if subject_path.exists():
            print(f"\n=== {subject_id} ===")
            
            try:
                with open(subject_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                labels = data['label']
                unique_labels = np.unique(labels)
                
                print(f"Unique labels: {unique_labels}")
                print(f"Label counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
                print(f"Total samples: {len(labels)}")
                
                # Check label meanings
                print("Label distribution over time (first 10000 samples):")
                for i in range(0, min(10000, len(labels)), 1000):
                    window = labels[i:i+1000]
                    unique_window = np.unique(window)
                    print(f"  Samples {i:5d}-{i+1000:5d}: {unique_window}")
                
            except Exception as e:
                print(f"Error loading {subject_id}: {e}")
        else:
            print(f"{subject_id} not found")

if __name__ == "__main__":
    inspect_wesad_labels()
