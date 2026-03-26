#!/usr/bin/env python3
"""Quick test of the fixed dataset adapter."""

import sys
import json
from pathlib import Path

# Add code/script to path
sys.path.insert(0, str(Path(__file__).parent / 'code/script'))

print("Testing Stage 4 Dataset Adapter")
print("=" * 50)

# First, check what's in a prepared file
prepared_dir = Path(__file__).parent / 'data/prepared/train'
print(f"Checking {prepared_dir}")

if prepared_dir.exists():
    json_files = sorted(prepared_dir.glob('*.json'))
    print(f"✓ Found {len(json_files)} prepared JSON files")
    
    # Load and preview first file
    with open(json_files[0]) as f:
        data = json.load(f)
    
    print(f"\n✓ First file keys: {list(data.keys())}")
    print(f"  - hash_name: {data['hash_name']}")
    print(f"  - image_width: {data['image_width']}")
    print(f"  - image_height: {data['image_height']}")
    print(f"  - num_words: {data['num_words']}")
    print(f"  - num_kvps: {data['num_kvps']}")
    
    # Preview lmdx_text
    lmdx_lines = data['lmdx_text'].split('\n')[:3]
    print(f"\n✓ lmdx_text (first 3 lines):")
    for line in lmdx_lines:
        print(f"    {line}")
    
    # Preview gt_kvps
    kvps = data.get('gt_kvps', {}).get('kvps_list', [])
    if kvps:
        print(f"\n✓ Found {len(kvps)} key-value pairs")
        print(f"  Example KVP: {kvps[0]}")
    
    print("\n" + "=" * 50)
    print("✅ Data format verified successfully!")
    print("\nNow attempting to load via dataset adapter...")
    
    try:
        from stage4_kvp_dataset import LayoutLMv3PreparedDataset
        
        dataset = LayoutLMv3PreparedDataset(
            data_dir=str(Path(__file__).parent / 'data/prepared'),
            split='train',
            include_images=False
        )
        
        print(f"✓ Dataset created, size: {len(dataset)}")
        
        # Try loading a sample
        item = dataset[0]
        print(f"\n✓ Sample loaded!")
        for key, val in item.items():
            if hasattr(val, 'shape'):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {type(val).__name__}")
        
        print("\n✅ SUCCESS: Dataset adapter is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error loading via dataset adapter:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ Prepared data directory not found: {prepared_dir}")
