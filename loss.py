import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePreferenceLoss(nn.Module):
    """Simple Preference Loss for PALC - no reference model or beta needed.

    Since PALC only trains the correction engine while keeping the base model frozen,
    we don't need the complex reference model constraints that DPO provides.
    This loss directly optimizes for preferred responses being more likely than
    rejected ones without any KL divergence regularization.
    """

    def __init__(self, pad_token_id=None):
        super().__init__()
        self.pad_token_id = pad_token_id

    def _get_log_probs(self, logits, labels, prompt_lengths, attention_mask=None):
        """Calculate log probabilities for response tokens only."""
        batch_size = logits.shape[0]

        # Shift for next-token prediction
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:].clone()

        # Handle padding tokens
        if self.pad_token_id is not None:
            labels_shifted[labels_shifted == self.pad_token_id] = -100

        batch_log_probs = []
        for i in range(batch_size):
            # Get valid positions (not padding)
            valid_mask = (labels_shifted[i] != -100)
            if not valid_mask.any():
                batch_log_probs.append(torch.tensor(0.0, device=logits.device))
                continue

            # Extract log probs for valid tokens
            valid_indices = torch.where(valid_mask)[0]
            valid_log_probs = log_probs[i, valid_indices]
            valid_labels = labels_shifted[i, valid_indices]

            # Get prompt length for this sample
            prompt_len = prompt_lengths[i].item() if torch.is_tensor(prompt_lengths[i]) else prompt_lengths[i]

            # Only compute loss on response tokens (after prompt)
            # Account for the shift: position 0 in shifted corresponds to position 1 in original
            response_start = max(0, prompt_len - 1)
            response_mask = (valid_indices >= response_start).float()

            if response_mask.sum() == 0:
                # No response tokens
                batch_log_probs.append(torch.tensor(0.0, device=logits.device))
                continue

            # Get log probs for response tokens only
            response_log_probs = valid_log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
            masked_log_probs = response_log_probs * response_mask

            # Average over response tokens
            avg_log_prob = masked_log_probs.sum() / response_mask.sum()
            batch_log_probs.append(avg_log_prob)

        return torch.stack(batch_log_probs)

    def forward(self, model, batch):
        """
        Forward pass calculating simple preference loss.

        Args:
            model: PALC model
            batch: Dictionary containing:
                - chosen_ids: Token IDs for chosen responses
                - rejected_ids: Token IDs for rejected responses
                - chosen_attention_mask: Attention mask for chosen
                - rejected_attention_mask: Attention mask for rejected
                - chosen_prompt_lengths: Prompt lengths for chosen
                - rejected_prompt_lengths: Prompt lengths for rejected
        """
        # Get model predictions for chosen and rejected responses
        chosen_logits = model(
            input_ids=batch["chosen_ids"],
            attention_mask=batch["chosen_mask"]
        )

        rejected_logits = model(
            input_ids=batch["rejected_ids"],
            attention_mask=batch["rejected_mask"]
        )

        # Calculate log probabilities for response portions only
        chosen_log_probs = self._get_log_probs(
            chosen_logits,
            batch["chosen_ids"],
            batch["prompt_lengths"],
            batch["chosen_mask"]
        )

        rejected_log_probs = self._get_log_probs(
            rejected_logits,
            batch["rejected_ids"],
            batch["prompt_lengths"],
            batch["rejected_mask"]
        )

        # Simple preference loss: chosen should be more likely than rejected
        # Direct optimization without beta scaling
        logits_diff = chosen_log_probs - rejected_log_probs
        loss = -F.logsigmoid(logits_diff).mean()

        # Additional metrics for monitoring
        with torch.no_grad():
            accuracy = (chosen_log_probs > rejected_log_probs).float().mean()
            chosen_mean = chosen_log_probs.mean()
            rejected_mean = rejected_log_probs.mean()
            margin = (chosen_log_probs - rejected_log_probs).mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "chosen_log_probs": chosen_mean,
            "rejected_log_probs": rejected_mean,
            "margin": margin
        }