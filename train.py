"""PALC training with HuggingFace Accelerate for distributed training."""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from model import create_palc_model
from data import PrefDataset, create_dataloader
from loss import SimplePreferenceLoss


def main():
    parser = argparse.ArgumentParser(description="Train PALC model with Accelerate")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--bottleneck_dim", type=int, default=32, help="Bottleneck dimension")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--max_length", type=int, default=1024)

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=2000)

    # Loss configuration
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")

    # Other arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16" if args.bf16 else "no"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer - try fast tokenizer first, fallback to slow if needed
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer failed ({e}), trying slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = PrefDataset(
        args.dataset_name, tokenizer, max_len=args.max_length, split="train", eval_split_ratio=0.1
    )
    eval_dataset = PrefDataset(
        args.dataset_name, tokenizer, max_len=args.max_length, split="eval", eval_split_ratio=0.1
    )

    # Create data loaders
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = create_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = create_palc_model(
        model_name=args.model_name,
        device=accelerator.device,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        bottleneck=args.bottleneck_dim
    )

    loss_fn = SimplePreferenceLoss(
        pad_token_id=tokenizer.pad_token_id
    )

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=args.learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, scheduler)

    global_step = 0
    if args.resume_from_checkpoint:
        if hasattr(model, 'module'):
            global_step, _, states_loaded = model.module.load_checkpoint(
                args.resume_from_checkpoint, optimizer, scheduler, accelerator.device
            )
        else:
            global_step, _, states_loaded = model.load_checkpoint(
                args.resume_from_checkpoint, optimizer, scheduler, accelerator.device
            )

    model.train()
    best_eval_loss = float('inf')
    last_eval_loss = None

    # Print debugging metrics guide
    if accelerator.is_main_process:
        print("=" * 80)
        print("PALC Training Debug Metrics Guide:")
        print("• loss: Training loss (lower is better)")
        print("• grad_norm: Gradient norm (0.001~0.1 = normal, >1.0 = explosion)")
        print("• prompt%: Prompt percentage (30-70% = normal range)")
        print("• lr: Learning rate (scheduled)")
        print("• eval: Evaluation loss (when available)")
        print("=" * 80)

    for epoch in range(args.num_epochs):
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc="Training", disable=not accelerator.is_main_process)

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # SimplePreferenceLoss handles model forward pass internally
                loss_results = loss_fn(model, batch)
                loss = loss_results["loss"]

                accelerator.backward(loss)

                # Calculate gradient norm before clipping/zeroing
                total_grad_norm = 0
                if accelerator.sync_gradients:
                    for param in model.get_trainable_parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm().item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    accelerator.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
                else:
                    total_grad_norm = 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1

                    # Move evaluation and checkpoint logic here to execute only once per global_step
                    if accelerator.is_main_process:
                        # Evaluation at specified intervals
                        if global_step > 0 and global_step % args.eval_steps == 0:
                            model.eval()
                            eval_loss_values = []
                            with torch.no_grad():
                                for eval_batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                                    eval_loss_results = loss_fn(model, eval_batch)
                                    eval_loss_values.append(eval_loss_results["loss"].item())

                            avg_eval_loss = np.mean(eval_loss_values)
                            last_eval_loss = avg_eval_loss
                            if avg_eval_loss < best_eval_loss:
                                best_eval_loss = avg_eval_loss
                                save_path = os.path.join(args.output_dir, "best_palc.pt")
                                if hasattr(model, 'module'):
                                    model.module.save_weights(save_path)
                                else:
                                    model.save_weights(save_path)
                            model.train()

                        # Checkpoint saving at specified intervals
                        if global_step > 0 and global_step % args.save_steps == 0:
                            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                            if hasattr(model, 'module'):
                                model.module.save_checkpoint(checkpoint_path, optimizer, scheduler, global_step, vars(args))
                            else:
                                model.save_checkpoint(checkpoint_path, optimizer, scheduler, global_step, vars(args))

                    # Record loss only when gradients are synchronized (accumulation step complete)
                    epoch_losses.append(loss.item())

                # Show progress (simplified metrics)
                if accelerator.is_main_process and step % 10 == 0:
                    # Calculate simple debugging metrics
                    with torch.no_grad():
                        # Prompt masking statistics
                        avg_prompt_len = batch["prompt_lengths"].float().mean().item()
                        avg_seq_len = batch["chosen_mask"].sum(dim=1).float().mean().item()
                        prompt_ratio = avg_prompt_len / avg_seq_len if avg_seq_len > 0 else 0.0

                    progress_metrics = {
                        'loss': f"{loss.item():.4f}",
                        'grad_norm': f"{total_grad_norm:.4f}",
                        'prompt%': f"{prompt_ratio*100:.1f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                    if last_eval_loss is not None:
                        progress_metrics['eval'] = f"{last_eval_loss:.4f}"
                    progress_bar.set_postfix(progress_metrics)

        avg_epoch_loss = np.mean(epoch_losses)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_palc.pt")
        if hasattr(model, 'module'):
            model.module.save_weights(final_path)
        else:
            model.save_weights(final_path)

        best_path = os.path.join(args.output_dir, "best_palc.pt")
        if not os.path.exists(best_path):
            if hasattr(model, 'module'):
                model.module.save_weights(best_path)
            else:
                model.save_weights(best_path)


if __name__ == "__main__":
    main()