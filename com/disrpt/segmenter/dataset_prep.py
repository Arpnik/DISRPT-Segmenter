from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from com.disrpt.segmenter.utils.combined_dataset import CombinedDataset
from com.disrpt.segmenter.utils.data_downloader import DatasetDownloader
from com.disrpt.segmenter.utils.data_preprocesser import EDUDataset
import numpy as np


def download_dataset(dataset_dir="dataset"):
    """Download multiple EDU datasets"""
    downloader = DatasetDownloader(dataset_dir)
    return downloader.download_corpus()


def load_datasets(tokenizer, dataset_dir="dataset", max_length=512):
    """
    Load train, dev, and test splits from all datasets in dataset_dir

    Returns:
        train_dataset, dev_dataset, test_dataset, dataset_tracking
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find all dataset folders
    dataset_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not dataset_folders:
        raise FileNotFoundError(f"No dataset folders found in {dataset_dir}")

    print("\n" + "=" * 70)
    print("Discovering datasets in:", dataset_dir)
    print("=" * 70)
    print(f"Found {len(dataset_folders)} dataset(s): {[d.name for d in dataset_folders]}")

    # Collect file paths by split
    train_files = []
    dev_files = []
    test_files = []

    for folder in dataset_folders:
        dataset_name = folder.name

        # Check for .conllu files
        train_file = folder / "train.conllu"
        dev_file = folder / "dev.conllu"
        test_file = folder / "test.conllu"

        if train_file.exists():
            train_files.append(train_file)
            print(f"  ✓ {dataset_name}/train.conllu")

        if dev_file.exists():
            dev_files.append(dev_file)
            print(f"  ✓ {dataset_name}/dev.conllu")

        if test_file.exists():
            test_files.append(test_file)
            print(f"  ✓ {dataset_name}/test.conllu")

    # Verify we have files
    if not train_files:
        raise FileNotFoundError("No train.conllu files found in any dataset folder")

    print("\n" + "=" * 70)
    print("Loading and combining datasets for EDU segmentation")
    print("=" * 70)
    print(f"Train files: {len(train_files)}")
    print(f"Dev files:   {len(dev_files)}")
    print(f"Test files:  {len(test_files)}")

    # Load and combine datasets with tracking
    all_train_examples = []
    all_dev_examples = []
    all_test_examples = []

    # Track which dataset each example came from
    train_tracking = {}
    dev_tracking = {}
    test_tracking = {}

    # Load train splits
    for train_file in train_files:
        dataset_name = train_file.parent.name
        print(f"\nLoading train: {dataset_name}/{train_file.name}")
        dataset = EDUDataset(train_file, tokenizer, max_length)

        start_idx = len(all_train_examples)
        all_train_examples.extend(dataset.examples)
        end_idx = len(all_train_examples)

        # Track indices for this dataset
        for idx in range(start_idx, end_idx):
            train_tracking[idx] = dataset_name

    # Load dev splits
    for dev_file in dev_files:
        dataset_name = dev_file.parent.name
        print(f"\nLoading dev: {dataset_name}/{dev_file.name}")
        dataset = EDUDataset(dev_file, tokenizer, max_length)

        start_idx = len(all_dev_examples)
        all_dev_examples.extend(dataset.examples)
        end_idx = len(all_dev_examples)

        for idx in range(start_idx, end_idx):
            dev_tracking[idx] = dataset_name

    # Load test splits
    for test_file in test_files:
        dataset_name = test_file.parent.name
        print(f"\nLoading test: {dataset_name}/{test_file.name}")
        dataset = EDUDataset(test_file, tokenizer, max_length)

        start_idx = len(all_test_examples)
        all_test_examples.extend(dataset.examples)
        end_idx = len(all_test_examples)

        for idx in range(start_idx, end_idx):
            test_tracking[idx] = dataset_name

    # Create combined datasets
    train_dataset = CombinedDataset(all_train_examples)
    dev_dataset = CombinedDataset(all_dev_examples)
    test_dataset = CombinedDataset(all_test_examples)

    print("\n" + "=" * 70)
    print("Dataset Loading Complete!")
    print("=" * 70)
    print(f"Train: {len(train_dataset)} examples (from {len(train_files)} file(s))")
    print(f"Dev:   {len(dev_dataset)} examples (from {len(dev_files)} file(s))")
    print(f"Test:  {len(test_dataset)} examples (from {len(test_files)} file(s))")
    print("=" * 70)

    dataset_tracking = {
        'train': train_tracking,
        'dev': dev_tracking,
        'test': test_tracking
    }

    return train_dataset, dev_dataset, test_dataset, dataset_tracking


if __name__ == "__main__":

    print("\n" + "=" * 35)
    print("Multi-Dataset EDU Segmentation Loader")
    print("=" * 35)

    download_success = download_dataset()

    if not download_success:
        print("\n❌ Download failed. Please check your internet connection.")
        exit(1)

    # Step 2: Load tokenizer
    print("\n" + "=" * 70)
    print("STEP 2: Load Tokenizer")
    print("=" * 70)
    print("Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("✓ Tokenizer loaded")

    # Step 3: Load datasets
    print("\n" + "=" * 70)
    print("STEP 3: Load and Process Datasets")
    print("=" * 70)
    train_dataset, dev_dataset, test_dataset, dataset_tracking = load_datasets(tokenizer)

    # Show example
    print("\n" + "=" * 70)
    print("Sample Data Point")
    print("=" * 70)
    example = train_dataset[0]
    print(f"Input IDs shape:      {example['input_ids'].shape}")
    print(f"Attention mask shape: {example['attention_mask'].shape}")
    print(f"Labels shape:         {example['labels'].shape}")
    print(f"\nFirst 20 tokens (IDs): {example['input_ids'][:20].tolist()}")
    print(f"First 20 labels:       {example['labels'][:20].tolist()}")
    print("\nLabel meanings:")
    print("  -100 = Ignore (special tokens or subword continuations)")
    print("     0 = Token continues current EDU")
    print("     1 = Token starts new EDU")

    print("\n" + "✅" * 35)
    print("Dataset ready for training!")
    print("✅" * 35)

    print("\n" + "=" * 70)
    print("VALIDATION: Detailed Token & Label Inspection")
    print("=" * 70)

    example = train_dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
    labels = example['labels']

    print("\nFirst 40 tokens with their labels:")
    print(f"{'Idx':<5} {'Token':<20} {'Label':<10} {'Meaning'}")
    print("-" * 60)

    for i in range(min(40, len(tokens))):
        label_val = labels[i].item()
        if label_val == -100:
            meaning = "IGNORE (special/subword)"
        elif label_val == 1:
            meaning = "EDU START ← New segment begins"
        else:
            meaning = "CONTINUE (within EDU)"

        print(f"{i:<5} {tokens[i]:<20} {label_val:<10} {meaning}")

    # Statistics
    print("\n" + "=" * 70)
    print("Label Distribution Statistics")
    print("=" * 70)

    total = len(labels)
    ignored = (labels == -100).sum().item()
    edu_starts = (labels == 1).sum().item()
    continues = (labels == 0).sum().item()

    print(f"Total tokens:           {total}")
    print(f"Ignored (-100):         {ignored:5d} ({ignored / total * 100:5.1f}%) [Special tokens + subwords]")
    print(f"EDU starts (1):         {edu_starts:5d} ({edu_starts / total * 100:5.1f}%) [New segments]")
    print(f"EDU continues (0):      {continues:5d} ({continues / total * 100:5.1f}%) [Within segments]")

    # Class balance check
    if edu_starts > 0:
        valid_labels = edu_starts + continues
        print(f"\nClass Balance (excluding ignored):")
        print(f"  Positive (EDU start):    {edu_starts / valid_labels * 100:.1f}%")
        print(f"  Negative (continue):     {continues / valid_labels * 100:.1f}%")
        print(f"  Imbalance ratio:         {continues / edu_starts:.1f}:1")

    # Verify shapes match
    print("\n" + "=" * 70)
    print("Tensor Shape Validation")
    print("=" * 70)
    print(f"input_ids shape:      {example['input_ids'].shape}")
    print(f"attention_mask shape: {example['attention_mask'].shape}")
    print(f"labels shape:         {example['labels'].shape}")
    assert example['input_ids'].shape == example['labels'].shape, "Shape mismatch!"
    print("✓ All shapes match correctly")

    print("\n" + "=" * 70)
    print("DATASET-WIDE ANALYSIS (Per Dataset)")
    print("=" * 70)

    # Analyze by dataset
    dataset_stats = {}
    train_tracking_dict = dataset_tracking['train']

    for i in range(len(train_dataset)):
        example = train_dataset[i]
        labels = example['labels']
        attention_mask = example['attention_mask']

        source = train_tracking_dict.get(i, 'unknown')

        if source not in dataset_stats:
            dataset_stats[source] = {
                'lengths': [],
                'edu_counts': [],
                'valid_tokens': []
            }

        real_tokens = attention_mask.sum().item()
        edu_starts = (labels == 1).sum().item()
        valid_labels = ((labels != -100) & (attention_mask == 1)).sum().item()

        dataset_stats[source]['lengths'].append(real_tokens)
        dataset_stats[source]['edu_counts'].append(edu_starts)
        dataset_stats[source]['valid_tokens'].append(valid_labels)

    # Print stats per dataset
    for dataset_name in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset_name]
        print(f"\n📊 {dataset_name.upper()} Dataset:")
        print(f"  Examples:             {len(stats['lengths'])}")
        print(f"  Mean length:          {np.mean(stats['lengths']):.1f} tokens")
        print(f"  Median length:        {np.median(stats['lengths']):.1f} tokens")
        print(f"  Min/Max length:       {np.min(stats['lengths'])}/{np.max(stats['lengths'])} tokens")
        print(f"  Mean EDUs/sequence:   {np.mean(stats['edu_counts']):.2f}")
        print(f"  Total EDUs:           {np.sum(stats['edu_counts'])}")
        print(f"  Mean valid tokens:    {np.mean(stats['valid_tokens']):.1f}")

    # Overall combined stats
    print(f"\n📊 COMBINED Dataset:")
    all_lengths = [l for s in dataset_stats.values() for l in s['lengths']]
    all_edus = [e for s in dataset_stats.values() for e in s['edu_counts']]
    all_valid = [v for s in dataset_stats.values() for v in s['valid_tokens']]

    print(f"  Total examples:       {len(all_lengths)}")
    print(f"  Mean length:          {np.mean(all_lengths):.1f} tokens")
    print(f"  Median length:        {np.median(all_lengths):.1f} tokens")
    print(f"  Min/Max length:       {np.min(all_lengths)}/{np.max(all_lengths)} tokens")
    print(f"  Mean EDUs/sequence:   {np.mean(all_edus):.2f}")
    print(f"  Total EDUs:           {np.sum(all_edus)}")
    print(f"  Mean valid tokens:    {np.mean(all_valid):.1f}")

    # Show sample sequences from each dataset
    print("\n" + "=" * 70)
    print("Sample Sequences (from each dataset):")
    print("=" * 70)

    # Get first example from each dataset
    shown_datasets = set()
    for i in range(len(train_dataset)):
        source = train_tracking_dict.get(i, 'unknown')

        if source not in shown_datasets:
            example = train_dataset[i]
            tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
            labels = example['labels']
            attention_mask = example['attention_mask']

            real_tokens = attention_mask.sum().item()
            edu_count = (labels == 1).sum().item()

            # Get actual text (excluding [CLS], [SEP], [PAD])
            text_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1 and t not in ['[CLS]', '[SEP]']]
            text = tokenizer.convert_tokens_to_string(text_tokens)

            print(f"\n[{source.upper()}]")
            print(f"  Length: {real_tokens} tokens | EDUs: {edu_count}")
            print(f"  Text: {text[:150]}...")

            shown_datasets.add(source)