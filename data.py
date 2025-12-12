"""Data processing for PALC preference learning."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import os
import pickle
import hashlib


class PrefDataset(Dataset):
    """Preference dataset for DPO training with caching."""

    def __init__(self, dataset_name, tokenizer, max_len=1024, split="train",
                 eval_split_ratio=0.1, cache_dir="/home/ubuntu/sanghyun/research/palc/.cache"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = cache_dir

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Generate cache key based on dataset name, tokenizer, max_len, split, and ratio
        cache_key = self._generate_cache_key(dataset_name, tokenizer, max_len, split, eval_split_ratio)
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

        # Try to load from cache first
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} cached examples")
            return

        # Load dataset with proper splitting - handle different dataset formats
        try:
            # First try to load with standard "train" split
            if split == "train_full":
                data = load_dataset(dataset_name, split="train")
            else:
                full_data = load_dataset(dataset_name, split="train")
                total_size = len(full_data)

                # Calculate split sizes
                eval_size = int(total_size * eval_split_ratio)
                train_size = total_size - eval_size

                if split == "train":
                    data = full_data.select(range(train_size))
                elif split == "eval":
                    eval_start = train_size
                    eval_end = min(total_size, train_size + min(eval_size, 1000))
                    data = full_data.select(range(eval_start, eval_end))
                else:
                    raise ValueError(f"Invalid split: {split}. Use 'train', 'eval', or 'train_full'")
        except ValueError as e:
            if "Unknown split" in str(e):
                # Handle datasets with non-standard splits (like HuggingFaceH4/ultrafeedback_binarized)
                print(f"Standard 'train' split not found. Trying alternative splits...")

                # For HuggingFaceH4/ultrafeedback_binarized, directly load train_prefs split
                print(f"Loading train_prefs split directly...")

                if split == "train_full":
                    data = load_dataset(dataset_name, split="train_prefs")
                else:
                    full_data = load_dataset(dataset_name, split="train_prefs")
                    total_size = len(full_data)

                    # Calculate split sizes
                    eval_size = int(total_size * eval_split_ratio)
                    train_size = total_size - eval_size

                    if split == "train":
                        data = full_data.select(range(train_size))
                    elif split == "eval":
                        eval_start = train_size
                        eval_end = min(total_size, train_size + min(eval_size, 1000))
                        data = full_data.select(range(eval_start, eval_end))
                    else:
                        raise ValueError(f"Invalid split: {split}. Use 'train', 'eval', or 'train_full'")
            else:
                raise

        # Process dataset (this will be cached)
        print(f"Processing {split} dataset (this may take a few minutes first time)")
        print(f"Total examples to process: {len(data)}")
        self.data = []
        processed_count = 0

        for i, example in enumerate(tqdm(data, desc=f"Processing {split}", ncols=80, leave=True, mininterval=1.0)):
            try:
                processed_example = self._process_example(example)
                if processed_example:
                    self.data.append(processed_example)
                    processed_count += 1

                # Print progress every 1000 examples
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i+1}/{len(data)} examples, {processed_count} valid")

            except Exception as e:
                continue

        # Save to cache for next time
        print(f"Saving processed dataset to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Cached {len(self.data)} processed examples")

    def _generate_cache_key(self, dataset_name, tokenizer, max_len, split, eval_split_ratio):
        """Generate a unique cache key based on dataset parameters"""
        # Include tokenizer model name and relevant parameters
        tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown_tokenizer')

        # Create a string that uniquely identifies this configuration
        config_str = f"{dataset_name}_{tokenizer_name}_{max_len}_{split}_{eval_split_ratio}"

        # Hash it to create a manageable filename
        return hashlib.md5(config_str.encode()).hexdigest()

    def _format_messages(self, prompt, messages):
        """Format message list into a single text (for ultrafeedback_binarized format)"""
        if not messages:
            return prompt

        # Extract the assistant response from the message list
        formatted_text = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'user':
                # For user messages, format as Human:
                formatted_text += f"Human: {content}\n\n"
            elif role == 'assistant':
                # For assistant messages, format as Assistant:
                formatted_text += f"Assistant: {content}"

        return formatted_text.strip()

    def _process_example(self, example):
        """Process a single preference example"""
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        if not (chosen and rejected):
            return None

        # Handle different dataset formats
        if isinstance(chosen, list) and isinstance(rejected, list):
            # HuggingFaceH4/ultrafeedback_binarized format - chosen/rejected are message lists
            chosen_text = self._format_messages(prompt, chosen)
            rejected_text = self._format_messages(prompt, rejected)
        elif "Human:" in prompt and "Assistant:" in prompt:
            # Dahoas format - prompt contains partial Assistant response
            # The chosen/rejected are continuations, not complete responses

            # For Dahoas format, chosen/rejected are always continuations
            # Direct concatenation preserves the natural flow
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
        else:
            # Standard format - add proper separator
            chosen_text = f"{prompt}\n\nAssistant: {chosen}" if prompt else chosen
            rejected_text = f"{prompt}\n\nAssistant: {rejected}" if prompt else rejected

        # Tokenize full texts with padding disabled for efficiency
        chosen_tokenized = self.tokenizer(
            chosen_text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True
        )
        rejected_tokenized = self.tokenizer(
            rejected_text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True
        )

        # Find prompt token boundary using string-based detection
        prompt_length = self._find_prompt_token_length(prompt, chosen_text, chosen_tokenized['input_ids'][0])

        return {
            'chosen_ids': chosen_tokenized['input_ids'].squeeze(0),
            'chosen_mask': chosen_tokenized['attention_mask'].squeeze(0),
            'rejected_ids': rejected_tokenized['input_ids'].squeeze(0),
            'rejected_mask': rejected_tokenized['attention_mask'].squeeze(0),
            'prompt_length': prompt_length  # String-based prompt boundary detection
        }

    def _find_prompt_token_length(self, prompt, full_text, token_ids):
        """Accurate tokenizer-based prompt length calculation for LLaMA."""
        if not prompt:
            return 0

        prompt_clean = prompt.strip()
        if not prompt_clean:
            return 0

        # For LLaMA models, find exact Assistant marker position
        try:
            # Validate inputs
            if not isinstance(token_ids, torch.Tensor) or len(token_ids) == 0:
                raise ValueError("Invalid token_ids input")

            # For Dahoas format: prompt includes partial Assistant response
            # We need to find where the prompt ends (before chosen/rejected continuation)
            if "Human:" in prompt_clean and "Assistant:" in prompt_clean:
                # Dahoas format: prompt contains the partial Assistant response
                # The boundary is exactly at the end of the prompt
                try:
                    prompt_tokens = self.tokenizer.encode(
                        prompt_clean,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=len(token_ids)
                    )

                    # Account for BOS token if present
                    if (len(token_ids) > 0 and
                        hasattr(self.tokenizer, 'bos_token_id') and
                        self.tokenizer.bos_token_id is not None and
                        token_ids[0] == self.tokenizer.bos_token_id):
                        prompt_length = len(prompt_tokens) + 1  # +1 for BOS
                    else:
                        prompt_length = len(prompt_tokens)

                    return self._validate_boundary_position(token_ids, prompt_length)

                except Exception:
                    pass  # Fall through to alternative methods

            # Method 1: Precise token-by-token search for Assistant marker (for other formats)
            assistant_marker = "Assistant:"
            max_search_tokens = min(len(token_ids), 300)  # Reasonable search limit

            # Find the exact token position where Assistant marker appears
            for i in range(1, max_search_tokens + 1):
                try:
                    decoded_partial = self.tokenizer.decode(token_ids[:i], skip_special_tokens=False)
                except Exception:
                    continue  # Skip problematic token sequences

                # Check if we've found the Assistant marker and moved past it
                if assistant_marker in decoded_partial:
                    # Verify boundary integrity
                    marker_positions = [pos for pos, part in enumerate(decoded_partial.split(assistant_marker)) if pos > 0]
                    if not marker_positions:
                        continue

                    # Find if there's content after the Assistant marker
                    after_assistant = decoded_partial.split(assistant_marker)[-1].strip()
                    if after_assistant and len(after_assistant) > 0:
                        # We've started the response, so this is our boundary
                        boundary_pos = max(i - 1, 0)
                        # Additional safety: ensure we're not cutting mid-word
                        return self._validate_boundary_position(token_ids, boundary_pos)
                    elif i == len(token_ids) - 1:
                        # Edge case: prompt ends exactly with Assistant:
                        return i

            # Method 2: Fallback - more accurate tokenizer-based calculation
            if "Human:" in full_text and "Assistant:" in full_text:
                # Reconstruct prompt with Assistant marker
                if full_text.strip().endswith("Assistant:"):
                    prompt_with_marker = prompt_clean + "\n\nAssistant:"
                else:
                    # Find the context up to Assistant:
                    parts = full_text.split("Assistant:")
                    if len(parts) >= 2:
                        prompt_with_marker = parts[0] + "Assistant:"
                    else:
                        prompt_with_marker = prompt_clean + "\n\nAssistant:"
            else:
                # Non-conversational format
                prompt_with_marker = prompt_clean + "\n\nAssistant:"

            # Tokenize the reconstructed prompt safely
            try:
                prompt_tokens = self.tokenizer.encode(
                    prompt_with_marker,
                    add_special_tokens=False,  # Important for LLaMA
                    truncation=True,
                    max_length=len(token_ids)  # Don't exceed original length
                )
            except Exception:
                raise ValueError("Tokenization failed in fallback method")

            # Account for BOS token if present in the full sequence
            if (len(token_ids) > 0 and
                hasattr(self.tokenizer, 'bos_token_id') and
                self.tokenizer.bos_token_id is not None and
                token_ids[0] == self.tokenizer.bos_token_id):
                prompt_length = len(prompt_tokens) + 1  # +1 for BOS
            else:
                prompt_length = len(prompt_tokens)

            # Enhanced safety bounds with quality checks
            validated_length = self._validate_boundary_position(token_ids, prompt_length)
            return validated_length

        except Exception as e:
            # Ultimate fallback with enhanced safety
            return self._safe_fallback_calculation(prompt_clean, full_text, token_ids)

    def _validate_boundary_position(self, token_ids, proposed_length):
        """Validate and adjust boundary position for safety."""
        total_length = len(token_ids)

        # Basic bounds checking
        if proposed_length <= 0:
            return 1
        if proposed_length >= total_length:
            return max(total_length // 2, 1)

        # Ensure reasonable prompt-to-response ratio (10% to 80%)
        ratio = proposed_length / total_length
        if ratio < 0.1:
            return max(int(total_length * 0.1), 1)
        elif ratio > 0.8:
            return int(total_length * 0.8)

        return proposed_length

    def _safe_fallback_calculation(self, prompt_clean, full_text, token_ids):
        """Safe fallback calculation with multiple heuristics."""
        total_length = len(token_ids)

        if not prompt_clean or not full_text:
            return max(total_length // 3, 1)  # Conservative estimate

        # Improved character ratio with LLaMA considerations
        prompt_ratio = len(prompt_clean) / len(full_text)

        # Apply safety adjustments
        if prompt_ratio < 0.1:
            prompt_ratio = 0.15  # Minimum reasonable prompt ratio
        elif prompt_ratio > 0.8:
            prompt_ratio = 0.7   # Maximum reasonable prompt ratio

        # Add buffer for tokenization differences (15% for LLaMA)
        estimated_length = int(total_length * prompt_ratio * 1.15)

        # Final safety bounds
        return min(max(estimated_length, 1), total_length // 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_statistics(self):
        """Get dataset statistics"""
        if not self.data:
            return {}

        chosen_lengths = [example['chosen_mask'].sum().item() for example in self.data]
        rejected_lengths = [example['rejected_mask'].sum().item() for example in self.data]

        return {
            'num_examples': len(self.data),
            'avg_chosen_length': sum(chosen_lengths) / len(chosen_lengths),
            'avg_rejected_length': sum(rejected_lengths) / len(rejected_lengths),
            'max_chosen_length': max(chosen_lengths),
            'max_rejected_length': max(rejected_lengths),
            'min_chosen_length': min(chosen_lengths),
            'min_rejected_length': min(rejected_lengths)
        }

    def verify_prompt_boundaries(self, num_samples=5):
        """Verify that prompt boundaries are correctly detected"""
        if not self.data:
            print("No data to verify")
            return

        print(f"\n{'='*60}")
        print("LLaMA Prompt Boundary Verification")
        print(f"{'='*60}")

        for i in range(min(num_samples, len(self.data))):
            example = self.data[i]
            token_ids = example['chosen_ids']
            prompt_length = example['prompt_length']

            # Decode full sequence
            full_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

            # Decode prompt portion
            prompt_tokens = token_ids[:prompt_length]
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)

            # Decode response portion
            response_tokens = token_ids[prompt_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False)

            print(f"\nSample {i+1}:")
            print(f"Prompt length: {prompt_length} tokens")
            print(f"Total length: {len(token_ids)} tokens")
            print(f"Prompt ratio: {prompt_length/len(token_ids):.3f}")

            print("\n--- Detected Prompt ---")
            print(repr(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text))

            print("\n--- Detected Response ---")
            print(repr(response_text[:200] + "..." if len(response_text) > 200 else response_text))

            # Check if boundary looks correct based on format
            if "Human:" in full_text and "Assistant:" in full_text:
                # Dahoas format: prompt includes partial Assistant response
                boundary_looks_correct = (
                    "Assistant:" in prompt_text and
                    response_text.strip() and
                    len(response_text.strip()) > 0
                )
            else:
                # Standard format: prompt should end with Assistant:
                boundary_looks_correct = (
                    "Assistant:" in prompt_text and
                    prompt_text.strip().endswith("Assistant:") and
                    response_text.strip() and
                    not response_text.startswith("Assistant:")
                )

            print(f"\nBoundary Status: {'✓ CORRECT' if boundary_looks_correct else '✗ SUSPICIOUS'}")
            print("-" * 60)


def create_collate_fn(tokenizer):
    """Create a collate function with proper pad token.

    Args:
        tokenizer: Tokenizer to get pad_token_id from

    Returns:
        Collate function for DataLoader
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch):
        """Pad sequences to same length for batching

        Pads all sequences (chosen AND rejected) to the same maximum length
        to prevent tensor size mismatch in loss calculation.

        Args:
            batch: List of dataset examples

        Returns:
            Dictionary with padded tensors ready for model input
        """
        chosen_ids = [b['chosen_ids'] for b in batch]
        rejected_ids = [b['rejected_ids'] for b in batch]
        chosen_mask_list = [b['chosen_mask'] for b in batch]
        rejected_mask_list = [b['rejected_mask'] for b in batch]
        prompt_lengths = [b['prompt_length'] for b in batch]

        # Find the maximum length across ALL sequences in the batch (chosen + rejected)
        max_len = max(t.size(0) for t in chosen_ids + rejected_ids)

        def pad_tensors(tensors, pad_val=0):
            # All tensors are padded to the single max_len found above
            padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.size(0)] = t
            return padded

        return {
            'chosen_ids': pad_tensors(chosen_ids, pad_val=pad_token_id),
            'chosen_mask': pad_tensors(chosen_mask_list, pad_val=0),
            'rejected_ids': pad_tensors(rejected_ids, pad_val=pad_token_id),
            'rejected_mask': pad_tensors(rejected_mask_list, pad_val=0),
            'prompt_lengths': torch.tensor(prompt_lengths, dtype=torch.long)  # 추가
        }

    return collate_fn


def create_dataloader(dataset: PrefDataset, batch_size: int = 2, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """Create DataLoader for preference dataset

    Args:
        dataset: PrefDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance ready for training
    """
    # Create collate function with proper pad token
    collate_fn = create_collate_fn(dataset.tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def load_preference_data(dataset_name: str, tokenizer: PreTrainedTokenizer,
                        batch_size: int = 2, max_len: int = 512,
                        shuffle: bool = True, split: str = "train",
                        eval_split_ratio: float = 0.1) -> tuple[PrefDataset, DataLoader]:
    """Convenience function to load preference data and create dataloader

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_len: Maximum sequence length
        shuffle: Whether to shuffle data
        split: Dataset split ("train", "eval", or "train_full")
        eval_split_ratio: Ratio for eval split

    Returns:
        (dataset, dataloader) tuple
    """
    dataset = PrefDataset(dataset_name, tokenizer, max_len, split, eval_split_ratio)
    dataloader = create_dataloader(dataset, batch_size, shuffle)

    return dataset, dataloader


class CustomPrefDataset(PrefDataset):
    """Custom preference dataset for non-standard data formats

    Extends PrefDataset to handle custom data structures and preprocessing.
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer,
                 max_len: int = 512, prompt_key: str = "prompt",
                 chosen_key: str = "chosen", rejected_key: str = "rejected"):
        """Initialize with custom data

        Args:
            data: List of preference examples
            tokenizer: Tokenizer instance
            max_len: Maximum sequence length
            prompt_key: Key for prompt in data
            chosen_key: Key for chosen response
            rejected_key: Key for rejected response
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.data = []
        for example in tqdm(data, desc="Processing custom data"):
            try:
                processed_example = self._process_custom_example(example)
                if processed_example:
                    self.data.append(processed_example)
            except Exception:
                continue


    def _process_custom_example(self, example):
        """Process a single custom example"""
        prompt = example.get(self.prompt_key, "")
        chosen = example.get(self.chosen_key, "")
        rejected = example.get(self.rejected_key, "")

        if not (chosen and rejected):
            return None

        # Add proper separator for custom datasets
        chosen_text = f"{prompt}\n\nAssistant: {chosen}" if prompt else chosen
        rejected_text = f"{prompt}\n\nAssistant: {rejected}" if prompt else rejected

        # Tokenize full texts
        chosen_tokenized = self.tokenizer(chosen_text, max_length=self.max_len, truncation=True, return_tensors="pt")
        rejected_tokenized = self.tokenizer(rejected_text, max_length=self.max_len, truncation=True, return_tensors="pt")

        # Find prompt token boundary using string-based detection
        if prompt:
            # For custom datasets, prompt boundary includes "Assistant: " marker
            prompt_with_marker = f"{prompt}\n\nAssistant: "
            prompt_length = self._find_prompt_token_length_custom(prompt_with_marker, chosen_text, chosen_tokenized['input_ids'][0])
        else:
            prompt_length = 0

        return {
            'chosen_ids': chosen_tokenized['input_ids'].squeeze(0),
            'chosen_mask': chosen_tokenized['attention_mask'].squeeze(0),
            'rejected_ids': rejected_tokenized['input_ids'].squeeze(0),
            'rejected_mask': rejected_tokenized['attention_mask'].squeeze(0),
            'prompt_length': prompt_length  # String-based prompt boundary detection
        }

    def _find_prompt_token_length_custom(self, prompt_with_marker, full_text, token_ids):
        """Find where prompt+marker ends in the tokenized sequence."""
        if not prompt_with_marker:
            return 0

        prompt_clean = prompt_with_marker.strip()
        if not prompt_clean:
            return 0

        # Decode tokens incrementally until we cover the prompt + "Assistant: " marker
        for i in range(1, len(token_ids)):  # Start from 1 to skip BOS token
            decoded = self.tokenizer.decode(token_ids[1:i+1], skip_special_tokens=True).strip()

            # Check if we've covered the entire prompt with marker
            if len(decoded) >= len(prompt_clean):
                # Check if we've covered the "Assistant:" marker
                if "Assistant:" in decoded:
                    return i
                # If we've gone too far, return the previous position
                if len(decoded) > len(prompt_clean) * 1.2:  # 20% tolerance
                    return max(i - 1, 1)

        # Fallback: use character ratio estimation
        prompt_ratio = len(prompt_clean) / len(full_text) if full_text else 0
        estimated_length = int(len(token_ids) * prompt_ratio)
        return min(max(estimated_length, 1), len(token_ids) // 2)  # Reasonable bounds