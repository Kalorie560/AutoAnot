#!/usr/bin/env python3
"""
Convert dataset.npz to dataset.json

This script reads dataset.npz and converts it to dataset.json format
with the specified arrays: ['waveforms', 'labels', 'fs', 'metric', 'auto_labels']
"""

import numpy as np
import json
import os

def convert_npz_to_json(npz_path="dataset.npz", json_path="dataset.json"):
    """
    Convert .npz file to .json file
    
    Args:
        npz_path (str): Path to the input .npz file
        json_path (str): Path to the output .json file
    """
    
    # Check if npz file exists
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file {npz_path} not found")
    
    # Load npz file
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    # Print available keys for debugging
    print(f"Available keys in {npz_path}: {list(data.keys())}")
    
    # Initialize output dictionary
    json_data = {}
    
    # Required arrays to include
    required_keys = ['waveforms', 'labels', 'fs', 'metric', 'auto_labels']
    
    for key in required_keys:
        if key in data:
            value = data[key]
            
            # Convert numpy arrays/values to JSON-serializable formats
            if isinstance(value, np.ndarray):
                if value.dtype.kind in ['U', 'S']:  # String arrays
                    json_data[key] = value.tolist()
                elif value.dtype == np.float32 or value.dtype == np.float64:
                    json_data[key] = value.tolist()
                elif value.dtype == np.int32 or value.dtype == np.int64:
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_data[key] = value.item()  # Convert numpy scalar to Python scalar
            else:
                json_data[key] = value  # Should be string or other JSON-serializable type
            
            print(f"✓ Added {key}: {type(json_data[key])}")
        else:
            print(f"⚠ Warning: Key '{key}' not found in dataset")
    
    # Save as JSON
    print(f"Saving to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully converted {npz_path} to {json_path}")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    for key, value in json_data.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):  # 2D array
                print(f"{key}: {len(value)}x{len(value[0])} array")
            else:  # 1D array
                print(f"{key}: {len(value)} elements")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    try:
        convert_npz_to_json()
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)