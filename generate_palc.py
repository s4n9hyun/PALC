#!/usr/bin/env python3
"""
Generate responses using PALC (Preference Alignment via Logit Calibration) model for evaluation.
PALC models use direct preference alignment through logit calibration.
Properly handles multi-turn conversations for MT-Bench.
"""

import sys
import json
import torch
import random
import numpy as np
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add PALC directory to path
sys.path.append('/home/elicer/research')
from palc.inference import load_palc


def load_dataset_by_name(dataset_name):
    """Load dataset based on dataset name."""
    if dataset_name == "hh_rlhf":
        print("Loading HH-RLHF test dataset...")
        dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
        prompt_key = "prompt"
        assistant_separator = ""
    elif dataset_name == "alpaca_eval":
        print("Loading AlpacaEval dataset from local file...")
        with open("/home/elicer/research/evaluation/alpaca_eval.json", "r") as f:
            alpaca_data = json.load(f)
        dataset = Dataset.from_list(alpaca_data)
        prompt_key = "instruction"
        assistant_separator = ""
    elif dataset_name == "mt_bench":
        print("Loading MT-Bench dataset...")
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompt_key = "prompt"
        assistant_separator = ""  # MT-Bench uses conversational format
    else:
        print(f"Unknown dataset: {dataset_name}, falling back to HH-RLHF")
        dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
        prompt_key = "prompt"
        assistant_separator = ""

    return dataset, prompt_key, assistant_separator


def sample_dataset(dataset, num_samples, random_seed=42):
    """Sample random indices from dataset."""
    random.seed(random_seed)
    all_indices = list(range(len(dataset)))
    sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
    sampled_data = dataset.select(sampled_indices)
    return sampled_data, sampled_indices


def process_mt_bench_sample(sample, prompt_key, palc, max_new_tokens, temperature, top_p):
    """Process MT-Bench sample with multi-turn conversation."""
    prompts_list = sample[prompt_key] if isinstance(sample[prompt_key], list) else [sample[prompt_key]]

    # Generate responses for all turns
    mt_bench_responses = []
    conversation_history = ""

    for turn_idx, question in enumerate(prompts_list):
        # Build conversation history
        if turn_idx == 0:
            turn_prompt = f"Human: {question}\n\nAssistant:"
        else:
            # Use previous conversation + new question
            turn_prompt = conversation_history + f"\n\nHuman: {question}\n\nAssistant:"

        # Generate response for this turn using PALC with sampling parameters
        full_response, turn_response = palc.generate(
            prompt=turn_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        mt_bench_responses.append({
            'turn': turn_idx + 1,
            'question': question,
            'response': turn_response
        })

        # Update conversation history for next turn
        conversation_history = turn_prompt + " " + turn_response

    # For compatibility with other scripts, use first turn as primary
    prompt = f"Human: {prompts_list[0]}\n\nAssistant:"
    generated_text = mt_bench_responses[0]['response']

    return prompt, generated_text, mt_bench_responses


def process_regular_sample(sample, prompt_key, dataset_name):
    """Process regular dataset samples (non-MT-Bench)."""
    if dataset_name == "hh_rlhf":
        prompt = sample[prompt_key]  # prompt field already formatted correctly
    elif dataset_name == "alpaca_eval":
        prompt = sample[prompt_key]
    else:
        # Fallback for unknown datasets
        prompt = sample.get(prompt_key, str(sample))

    return prompt


def build_result_dict(sample_id, prompt, generated_text, dataset_name, sample, checkpoint_path, base_model, max_new_tokens,
                      calibration_scale, temperature, top_p, mt_bench_responses=None):
    """Build result dictionary for a sample."""
    result = {
        'sample_id': sample_id,
        'prompt': prompt,
        'response': generated_text,
        'method': 'PALC',
        'dataset': dataset_name,
        'max_new_tokens': max_new_tokens,
        'palc_checkpoint': checkpoint_path,
        'model_path': base_model,
        'calibration_scale': calibration_scale,
        'temperature': temperature,
        'top_p': top_p,
    }

    # Add MT-Bench specific data
    if mt_bench_responses is not None:
        result.update({
            'mt_bench_turns': mt_bench_responses,
            'prompt_id': sample.get('prompt_id', 'unknown'),
            'category': sample.get('category', 'unknown'),
            'full_prompt': sample.get('prompt', 'unknown'),
        })

    return result


def generate_palc_responses(num_samples=300, dataset_name="hh_rlhf", random_seed=42,
                           max_new_tokens=1024, checkpoint_path=None,
                           base_model_name=None, bottleneck=64,
                           calibration_scale=5.0, temperature=0.5, top_p=0.9):
    """Generate responses using PALC model with sampling parameters."""

    print("=== PALC Model Response Generation ===")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Configuration
    if base_model_name is None:
        base_model = "argsearch/llama-7b-sft-float32"
    else:
        base_model = base_model_name
    print(f"Using base model: {base_model}")

    if checkpoint_path is None:
        checkpoint_path = "/home/elicer/research/palc/outputs/run_20250921_191405/final_palc.pt"
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Sampling config: calibration_scale={calibration_scale}, temperature={temperature}, top_p={top_p}")

    try:
        # Determine device
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = "cuda:1"  # Use GPU 1
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        print(f"Using device: {device}")

        # Load PALC model with simplified interface
        print("Loading PALC model...")
        print(f"Model config: bottleneck={bottleneck}, calibration_scale={calibration_scale}")
        palc = load_palc(
            model_path=checkpoint_path,
            model_name=base_model,
            device=device,
            bottleneck=bottleneck,
            calibration_scale=calibration_scale
        )
        print("PALC model loaded successfully!")

        # Load dataset
        dataset, prompt_key, assistant_separator = load_dataset_by_name(dataset_name)

        # Sample random indices
        sampled_data, sampled_indices = sample_dataset(dataset, num_samples, random_seed)

        print(f"Generating responses for {len(sampled_indices)} samples...")

        results = []

        for i, sample in enumerate(tqdm(sampled_data, desc="Generating PALC responses")):
            try:
                # Process sample based on dataset type
                if dataset_name == "mt_bench":
                    prompt, generated_text, mt_bench_responses = process_mt_bench_sample(
                        sample, prompt_key, palc, max_new_tokens, temperature, top_p
                    )
                    result = build_result_dict(
                        sampled_indices[i], prompt, generated_text, dataset_name, sample,
                        checkpoint_path, base_model, max_new_tokens,
                        calibration_scale, temperature, top_p, mt_bench_responses
                    )
                else:
                    # Process regular datasets
                    prompt = process_regular_sample(sample, prompt_key, dataset_name)

                    # Generate response using PALC with sampling parameters
                    full_response, generated_text = palc.generate(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

                    result = build_result_dict(
                        sampled_indices[i], prompt, generated_text, dataset_name, sample,
                        checkpoint_path, base_model, max_new_tokens,
                        calibration_scale, temperature, top_p
                    )

                results.append(result)

                # Clear GPU memory periodically
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': 'PALC',
                    'dataset': dataset_name,
                    'error': str(e)
                })

        # Save results
        # Save to dataset-specific subdirectory
        output_dir = f"/home/elicer/research/evaluation/outputs/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename based on checkpoint and parameters
        if checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
            output_file = f"{output_dir}/palc_{checkpoint_name}_{dataset_name}_scale{calibration_scale}_temp{temperature}_topp{top_p}_{num_samples}samples.json"
        else:
            output_file = f"{output_dir}/palc_{dataset_name}_scale{calibration_scale}_temp{temperature}_topp{top_p}_{num_samples}samples.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"PALC responses saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PALC model responses for evaluation")
    parser.add_argument("--dataset", type=str, default="hh_rlhf",
                       choices=["hh_rlhf", "alpaca_eval", "mt_bench"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--checkpoint", type=str, default="/home/elicer/research/palc/outputs/main_run_20250922_005203/final_palc.pt",
                       help="Path to PALC checkpoint")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (default: argsearch/llama-7b-sft-float32)")
    parser.add_argument("--bottleneck", type=int, default=256,
                       help="Bottleneck dimension (must match training config)")
    parser.add_argument("--calibration_scale", type=float, default=1.0,
                       help="Correction scale factor (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling threshold (default: 0.9)")
    parser.add_argument("num_samples", type=int, nargs="?", default=300, help="Number of samples to generate")

    args = parser.parse_args()
    print(f"Debug: num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}, dataset={args.dataset}")
    print(f"Debug: calibration_scale={args.calibration_scale}, temperature={args.temperature}, top_p={args.top_p}")
    generate_palc_responses(args.num_samples, args.dataset, args.seed, args.max_new_tokens,
                           args.checkpoint, args.base_model, args.bottleneck,
                           args.calibration_scale, args.temperature, args.top_p)