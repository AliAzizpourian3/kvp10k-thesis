"""
Data loading utilities for KVP10k dataset.
Handles dataset loading, filtering, and sampling.
"""

from datasets import load_dataset
from tqdm import tqdm
import config


def load_kvp10k_dataset(split="train", cache_dir=None):
    """
    Load KVP10k dataset from HuggingFace.
    
    Args:
        split: Dataset split ('train' or 'test')
        cache_dir: Cache directory for dataset
        
    Returns:
        Dataset object
    """
    if cache_dir is None:
        cache_dir = config.KVP_CACHE
        
    print(f"Loading KVP10k {split} split from HuggingFace...")
    dataset = load_dataset(
        config.DATASET_NAME,
        split=split,
        cache_dir=cache_dir
    )
    print(f"Loaded {len(dataset)} samples")
    return dataset


def filter_annotated_pages(dataset):
    """
    Filter dataset to only include pages with annotations.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        List of indices with annotations
    """
    print("Filtering for pages with annotations...")
    valid_indices = []
    
    for i, example in enumerate(tqdm(dataset)):
        annotations = example.get('annotations', [])
        if annotations and len(annotations) > 0:
            valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} pages with annotations out of {len(dataset)}")
    return valid_indices


def create_subset(dataset, n_samples, filter_annotated=True, seed=None):
    """
    Create a subset of the dataset for prototyping.
    
    Args:
        dataset: Full dataset
        n_samples: Number of samples to select
        filter_annotated: Only select pages with annotations
        seed: Random seed for reproducibility
        
    Returns:
        Dataset subset
    """
    if seed is None:
        seed = config.RANDOM_SEED

    # If n_samples is None, interpret as "use full dataset" (or full annotated subset)
    # This keeps Stage 1 compatible with SUBSET_N=None for full runs.
    if n_samples is None:
        if filter_annotated:
            valid_indices = filter_annotated_pages(dataset)
            return dataset.select(valid_indices)
        return dataset
    
    if filter_annotated:
        valid_indices = filter_annotated_pages(dataset)
        
        # Sample from valid indices
        import random
        random.seed(seed)
        
        if len(valid_indices) < n_samples:
            print(f"Warning: Only {len(valid_indices)} annotated pages available, using all")
            selected_indices = valid_indices
        else:
            selected_indices = random.sample(valid_indices, n_samples)
    else:
        # Random sample from full dataset
        selected_indices = list(range(len(dataset)))
        import random
        random.seed(seed)
        selected_indices = random.sample(selected_indices, min(n_samples, len(dataset)))
    
    subset = dataset.select(selected_indices)
    print(f"Created subset with {len(subset)} samples")
    return subset


def get_annotation_statistics(dataset):
    """
    Compute statistics about annotations in dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(dataset),
        'samples_with_annotations': 0,
        'total_annotations': 0,
        'annotations_per_sample': [],
        'label_types': set(),
        'linking_count': 0
    }
    
    for example in tqdm(dataset, desc="Computing annotation statistics"):
        annotations = example.get('annotations', [])
        
        if annotations and len(annotations) > 0:
            stats['samples_with_annotations'] += 1
            stats['annotations_per_sample'].append(len(annotations))
            stats['total_annotations'] += len(annotations)
            
            # Collect label types
            for ann in annotations:
                # The dataset schema may vary across HF versions:
                # - annotation can be a dict (old)
                # - annotation can be a string label (newer)
                if isinstance(ann, str):
                    if ann:
                        stats['label_types'].add(ann)
                    continue

                if not isinstance(ann, dict):
                    continue

                label = None
                label_field = ann.get('label')
                if isinstance(label_field, dict):
                    label = label_field.get('value') or label_field.get('text')
                elif isinstance(label_field, str):
                    label = label_field

                if label:
                    stats['label_types'].add(label)

                # Check for linking (only present in dict-style annotations)
                attributes = ann.get('attributes') or {}
                linking = None
                if isinstance(attributes, dict):
                    linking_field = attributes.get('Linking')
                    if isinstance(linking_field, dict):
                        linking = linking_field.get('value')
                    elif isinstance(linking_field, str):
                        linking = linking_field
                if linking:
                    stats['linking_count'] += 1
    
    # Compute averages
    if stats['annotations_per_sample']:
        stats['avg_annotations_per_sample'] = sum(stats['annotations_per_sample']) / len(stats['annotations_per_sample'])
    else:
        stats['avg_annotations_per_sample'] = 0
    
    return stats


if __name__ == "__main__":
    # Test data loading
    print("Testing data_loader.py...")
    
    # Load train split
    dataset = load_kvp10k_dataset(split="train")
    
    # Create subset
    subset = create_subset(dataset, n_samples=200, filter_annotated=True)
    
    # Get statistics
    stats = get_annotation_statistics(subset)
    
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with annotations: {stats['samples_with_annotations']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Average annotations per sample: {stats['avg_annotations_per_sample']:.2f}")
    print(f"Unique label types: {len(stats['label_types'])}")
    print(f"Annotations with linking: {stats['linking_count']}")
