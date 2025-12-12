#!/usr/bin/env python3
"""
PALC (Preference Alignment via Logit Calibration) Inference Time Measurement
Measures time to generate 128 tokens using PALC calibration
"""

import torch
import torch.nn.functional as F
import time
import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add PALC directory to path
sys.path.append('/home/ubuntu/sanghyun/research')
from palc.inference import load_palc


def force_generate_exact_tokens(palc_inference, prompt, num_tokens=128, temperature=1.0, top_p=0.9):
    """Generate exactly num_tokens by modifying the generation loop to ignore EOS"""
    model = palc_inference.model
    tokenizer = palc_inference.tokenizer
    device = palc_inference.device

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    original_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        for step in range(num_tokens):  # Force exactly num_tokens iterations
            calibrated_logits = model(inputs['input_ids'], inputs.get('attention_mask'))

            # Apply temperature (prevent division by zero)
            if temperature <= 0:
                # Use greedy decoding for temperature=0
                logits = calibrated_logits[:, -1, :]
            else:
                logits = calibrated_logits[:, -1, :] / temperature

            # Apply top-p (nucleus) filtering
            if top_p is not None and temperature > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            if (temperature <= 0 or temperature == 1.0) and top_p is None:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Update inputs - NO EOS CHECK, force continuation
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)

            if 'attention_mask' in inputs:
                attention_ones = torch.ones((inputs['attention_mask'].shape[0], 1),
                                          dtype=inputs['attention_mask'].dtype,
                                          device=inputs['attention_mask'].device)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_ones], dim=-1)

    full_response = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    generated_part = tokenizer.decode(inputs['input_ids'][0][original_length:], skip_special_tokens=True)

    return full_response, generated_part


def measure_inference_time(palc, prompt, num_tokens=128, num_runs=10, warmup_runs=2,
                         temperature=1.0, top_p=0.9):
    """Measure inference time for generating EXACTLY num_tokens with PALC"""

    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        # Use custom force generation to ensure exactly num_tokens
        _ = force_generate_exact_tokens(
            palc_inference=palc,
            prompt=prompt,
            num_tokens=num_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else None
        )

    # Actual measurement
    times = []
    generated_token_counts = []
    print(f"Running {num_runs} timed iterations...")
    for _ in tqdm(range(num_runs)):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.perf_counter()

        # Use custom force generation to ensure exactly num_tokens
        full_response, generated_text = force_generate_exact_tokens(
            palc_inference=palc,
            prompt=prompt,
            num_tokens=num_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else None
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        generation_time = end_time - start_time
        times.append(generation_time)

        # Count actual generated tokens using tokenizer
        generated_tokens = palc.tokenizer.encode(generated_text, add_special_tokens=False)
        actual_token_count = len(generated_tokens)
        generated_token_counts.append(actual_token_count)

        if actual_token_count != num_tokens:
            print(f"Warning: Generated {actual_token_count} tokens instead of {num_tokens}")

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "median_time": np.median(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "all_times": times,
        "tokens_per_second": num_tokens / np.mean(times),
        "generated_token_counts": generated_token_counts,
        "mean_generated_tokens": np.mean(generated_token_counts),
        "expected_tokens": num_tokens
    }


def main():
    parser = argparse.ArgumentParser(description="Measure PALC inference time")
    parser.add_argument("--model_name", type=str, default="argsearch/llama-7b-sft-float32",
                       help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str,
                       default="/home/ubuntu/sanghyun/research/palc/outputs/run_20250922_005203/final_palc.pt",
                       help="Path to PALC checkpoint")
    parser.add_argument("--calibration_scale", type=float, default=1.0,
                       help="Calibration scale for PALC")
    parser.add_argument("--bottleneck", type=int, default=256,
                       help="Bottleneck dimension for PALC")
    parser.add_argument("--num_tokens", type=int, default=128,
                       help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=10,
                       help="Number of runs for timing")
    parser.add_argument("--warmup_runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--prompt", type=str,
                       default="Human: What are the key benefits of regular exercise? Assistant:",
                       help="Prompt to use for generation")
    parser.add_argument("--output_file", type=str, default="results/palc_times.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use (default cuda:1 to match evaluation scripts)")

    args = parser.parse_args()

    print(f"Loading PALC model: {args.model_name}")
    print(f"Using checkpoint: {args.checkpoint_path}")
    print(f"Calibration scale: {args.calibration_scale}")
    print(f"Bottleneck dimension: {args.bottleneck}")
    print(f"Sampling config: temperature={args.temperature}, top_p={args.top_p}")

    # Determine device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"  # Use GPU 1
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load PALC model using the actual load_palc function
    print("Loading PALC model...")
    try:
        palc = load_palc(
            model_path=args.checkpoint_path,
            model_name=args.model_name,
            device=device,
            bottleneck=args.bottleneck,
            calibration_scale=args.calibration_scale
        )
        print("PALC model loaded successfully!")
    except Exception as e:
        print(f"Error loading PALC model: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nModel loaded. Starting inference time measurement...")
    print(f"Generating {args.num_tokens} tokens")
    print(f"Number of runs: {args.num_runs}")
    print(f"Prompt: {args.prompt[:100]}...")

    # Measure inference time
    results = measure_inference_time(
        palc=palc,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Add metadata
    results["metadata"] = {
        "model": args.model_name,
        "method": "PALC",
        "checkpoint_path": args.checkpoint_path,
        "calibration_scale": args.calibration_scale,
        "bottleneck": args.bottleneck,
        "num_tokens": args.num_tokens,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": str(device)
    }

    # Print results
    print("\n" + "="*50)
    print("RESULTS - PALC")
    print("="*50)
    print(f"Mean time: {results['mean_time']:.3f} seconds")
    print(f"Std dev: {results['std_time']:.3f} seconds")
    print(f"Median time: {results['median_time']:.3f} seconds")
    print(f"Min time: {results['min_time']:.3f} seconds")
    print(f"Max time: {results['max_time']:.3f} seconds")
    print(f"Tokens/second: {results['tokens_per_second']:.2f}")
    print("="*50)

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()