from torch.utils.data import Dataset
import torch

class EDUDataset(Dataset):
    """
    Dataset loader for eng.erst.gum EDU segmentation task.

    Reads CONLLU format and extracts:
    - Tokens (column 1)
    - Seg labels from MISC field (column 9): Seg=B-Seg or Seg=O

    Converts to binary classification:
    - Label 1: Token starts new EDU (Seg=B-Seg)
    - Label 0: Token continues current EDU (Seg=O)
    """

    def __init__(self, conllu_file, tokenizer, max_length=512, min_chunk_size=50):
        """
        Args:
            conllu_file: Path to .conllu file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (512 for BERT)
            min_chunk_size: Minimum tokens per training example
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"\n📚 Loading dataset from: {conllu_file}")

        # Read and combine sentences into chunks
        # Leave room for [CLS] and [SEP] tokens
        tokens_list, labels_list = self.read_conllu(
            conllu_file,
            min_tokens=min_chunk_size,
            max_tokens=max_length - 10  # Safety margin for special tokens
        )

        # Tokenize and align labels
        print(f"   Tokenizing {len(tokens_list)} document chunks...")
        for i, (tokens, labels) in enumerate(zip(tokens_list, labels_list)):
            encoding = self.tokenize_and_align_labels(tokens, labels)
            if encoding:
                self.examples.append(encoding)

            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(tokens_list)} chunks...")

        print(f"✓ Loaded {len(self.examples)} training examples")

    def read_conllu(self, conllu_file, min_tokens=50, max_tokens=400):
        """
        Read CONLLU and combine sentences into longer document chunks.

        Args:
            conllu_file: Path to .conllu file
            min_tokens: Minimum tokens per chunk (default 50)
            max_tokens: Maximum tokens per chunk (default 400, leaves room for [CLS]/[SEP])
        """
        # Step 1: Parse all individual sentences
        all_sentences = []

        with open(conllu_file, 'r', encoding='utf-8') as f:
            current_tokens = []
            current_labels = []

            for line in f:
                line = line.strip()

                # Empty line = end of sentence
                if not line:
                    if current_tokens:
                        all_sentences.append((current_tokens, current_labels))
                        current_tokens = []
                        current_labels = []
                    continue

                # Skip comments
                if line.startswith('#'):
                    continue

                # Parse token line
                parts = line.split('\t')
                if len(parts) >= 10:
                    token_id = parts[0]

                    # Skip multi-word tokens and empty nodes
                    if '-' in token_id or '.' in token_id:
                        continue

                    token = parts[1]
                    misc = parts[9]

                    # Extract Seg label
                    is_edu_start = 'Seg=B-Seg' in misc or 'Seg=B-seg' in misc

                    current_tokens.append(token)
                    current_labels.append(1 if is_edu_start else 0)

            # Last sentence
            if current_tokens:
                all_sentences.append((current_tokens, current_labels))

        print(f"   Parsed {len(all_sentences)} individual sentences")

        # Step 2: Combine sentences into document chunks
        chunks_tokens = []
        chunks_labels = []

        current_chunk_tokens = []
        current_chunk_labels = []

        for sent_tokens, sent_labels in all_sentences:
            # Would adding this sentence exceed max_tokens?
            future_length = len(current_chunk_tokens) + len(sent_tokens)

            if future_length > max_tokens and len(current_chunk_tokens) >= min_tokens:
                # Save current chunk and start new one
                chunks_tokens.append(current_chunk_tokens)
                chunks_labels.append(current_chunk_labels)

                current_chunk_tokens = sent_tokens.copy()
                current_chunk_labels = sent_labels.copy()
            else:
                # Add sentence to current chunk
                current_chunk_tokens.extend(sent_tokens)
                current_chunk_labels.extend(sent_labels)

        # Don't forget the last chunk
        if current_chunk_tokens and len(current_chunk_tokens) >= min_tokens:
            chunks_tokens.append(current_chunk_tokens)
            chunks_labels.append(current_chunk_labels)

        print(f"   Combined into {len(chunks_tokens)} document chunks")
        print(f"   Avg tokens per chunk: {sum(len(t) for t in chunks_tokens) / len(chunks_tokens):.1f}")
        print(f"   Avg EDUs per chunk: {sum(sum(l) for l in chunks_labels) / len(chunks_labels):.1f}")

        return chunks_tokens, chunks_labels


    def tokenize_and_align_labels(self, tokens, labels):
        """
        Tokenize with BERT/DistilBERT and align labels with subword tokens.

        Challenge: BERT splits words into subwords (e.g., "studying" → ["stud", "##ying"])
        Solution: Only predict on first subword, ignore others with label -100

        Example:
            Input:  ["studying", "ants"]  with labels [0, 1]
            Output: ["[CLS]", "stud", "##ying", "ant", "##s", "[SEP]"]
                    with labels [-100, 0, -100, 1, -100, -100]
        """
        if not tokens or not labels or len(tokens) != len(labels):
            return None

        try:
            # Tokenize with Hugging Face tokenizer
            tokenized = self.tokenizer(
                tokens,
                is_split_into_words=True,  # Input is pre-tokenized
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            # Align labels with subword tokens
            word_ids = tokenized.word_ids()  # Maps subwords to original word indices
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP], [PAD])
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word → use original label
                    aligned_labels.append(labels[word_idx])
                else:
                    # Subsequent subwords (##ying, ##s) → ignore in loss
                    aligned_labels.append(-100)

                previous_word_idx = word_idx

            return {
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0),
                'labels': torch.tensor(aligned_labels, dtype=torch.long)
            }

        except Exception as e:
            print(f"⚠️  Error tokenizing sequence: {e}")
            return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
