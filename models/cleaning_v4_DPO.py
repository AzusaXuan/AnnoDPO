import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import h5py
from tqdm import tqdm
import os
import sys
import wandb
import math
from typing import Optional, Tuple
import random
from torch.distributions.bernoulli import Bernoulli
from collections import defaultdict
import itertools
import gc  # Add import garbage collection module
from utils import all_gather_batch
import time
class DPOLoss(nn.Module):
    """
    DPO Loss - Support one preferred sample corresponding to multiple rejected samples, and can ignore invalid samples
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,  # [batch_size]
        policy_rejected_logps: torch.Tensor,  # [n_augments, batch_size]
        reference_chosen_logps: torch.Tensor,  # [batch_size]
        reference_rejected_logps: torch.Tensor,  # [n_augments, batch_size]
        rejected_weights: Optional[torch.Tensor] = None,  # [n_augments, batch_size] Sample validity weights
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get number of rejected samples
        n_augments = policy_rejected_logps.shape[0]
        
        # Expand preferred sample shape to match rejected samples [batch_size] -> [n_augments, batch_size]
        policy_chosen_logps_expanded = policy_chosen_logps.unsqueeze(0).expand(n_augments, -1)
        reference_chosen_logps_expanded = reference_chosen_logps.unsqueeze(0).expand(n_augments, -1)
            
        # Compute log ratio of each rejected sample relative to preferred sample
        pi_logratios = policy_chosen_logps_expanded - policy_rejected_logps  # [n_augments, batch_size]
        ref_logratios = reference_chosen_logps_expanded - reference_rejected_logps  # [n_augments, batch_size]
        logits = pi_logratios - ref_logratios  # [n_augments, batch_size]

        if self.ipo:
            # Compute IPO loss
            losses = (logits - 1 / (2 * self.beta)) ** 2  # [n_augments, batch_size]
        else:
            # Standard DPO loss calculation
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )  # [n_augments, batch_size]

        # Apply sample weights (if provided) - ignore invalid samples
        if rejected_weights is not None:
            # Multiply loss by weights
            losses = losses * rejected_weights  # [n_augments, batch_size]
            # Compute number of valid samples for normalization
            num_valid = rejected_weights.sum() + 1e-8  # Prevent division by zero
            loss = losses.sum() / num_valid
        else:
            # Original average calculation
            loss = losses.mean()
        
        # Compute reward (still keep original shape for compatibility with other code)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()  # [batch_size]
        
        # When averaging rejected sample rewards, also consider weights
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()  # [n_augments, batch_size]
        if rejected_weights is not None:
            # Weighted average
            rejected_rewards_mean = (rejected_rewards * rejected_weights).sum(dim=0) / (rejected_weights.sum(dim=0) + 1e-8)
        else:
            # Simple average
            rejected_rewards_mean = rejected_rewards.mean(dim=0)  # [batch_size]

        return loss, chosen_rewards, rejected_rewards_mean

class DPOTrainer:
    def __init__(self, 
                 policy_model,
                 reference_model,
                 beta=0.1, # 0.1
                 mask_ratio=0.8,
                 precision_penalty=0.5):  # Add precision penalty coefficient
        # self.policy = policy_model
        # self.reference_model = reference_model
        self.policy_module = policy_model
        self.reference_module = reference_model
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.label_smoothing = 0.01
        self.ipo = False
        self.precision_penalty = precision_penalty  # Store penalty coefficient
        # self.policy_module = self.policy.module if hasattr(self.policy, 'module') else self.policy
        # self.reference_module = self.reference_model.module if hasattr(self.reference_model, 'module') else self.reference_model
        self.loss_fn = DPOLoss(self.beta, self.label_smoothing, self.ipo)
        self.eval_count = 0
        self.reject_mode = "predict"

    def create_rejected_samples(self, orig_anno, n_augments):
        """
        Create more diverse rejected samples through multiple noise strategies - efficient vectorized implementation
        Args:
            orig_anno: [batch_size, n_anno] Original annotation
            n_augments: Number of noise versions created for each sample
        Returns:
            noisy_annos: [n_augments, batch_size, n_anno] Noisy annotation
        """
        batch_size, n_anno = orig_anno.shape
        device = orig_anno.device
        
        # Basic mask: identify non-zero (positive) and zero (negative) positions in original annotation
        positive_mask = (orig_anno != 0).float()  # [batch_size, n_anno]
        negative_mask = (orig_anno == 0).float()  # [batch_size, n_anno]
        
        # Expand original annotation to enhanced dimension
        expanded_anno = orig_anno.unsqueeze(0).expand(n_augments, -1, -1)  # [n_augments, batch_size, n_anno]
        expanded_pos_mask = positive_mask.unsqueeze(0).expand(n_augments, -1, -1)  # [n_augments, batch_size, n_anno]
        expanded_neg_mask = negative_mask.unsqueeze(0).expand(n_augments, -1, -1)  # [n_augments, batch_size, n_anno]
        
        # Randomly assign strategies to each enhanced sample (0-2 three strategies)
        # Strategy 2: Only add false positive labels, effect: precision increases slightly, but still decreases, recall decreases; Strategy 1: Only delete some real labels, effect: precision decreases, recall increases
        # All 2 is not allowed
        strategy_weights = torch.tensor([0.2, 0.8, 0.0], device=device)
        strategy_ids = torch.multinomial(strategy_weights, n_augments, replacement=True)
        
        # Prepare masks for all strategies
        strategy_masks = torch.zeros(3, n_augments, 1, 1, device=device)
        for i in range(3):
            strategy_masks[i, strategy_ids == i] = 1.0
        
        # Generate random tensors required for all strategies
        rand_tensor = torch.rand(n_augments, batch_size, n_anno, device=device)
        
        # --- Strategy 1: Only delete some real labels ---
        # Use fixed deletion rate instead of randomly generated deletion rate
        fixed_mask_ratio = 0.1  # Fixed retention rate of 80%, i.e., delete 20% of real labels
        rand_tensor = torch.rand(n_augments, batch_size, n_anno, device=device)
        deletion_masks = (rand_tensor < fixed_mask_ratio).float()  # [n_augments, batch_size, n_anno]
        strategy1_out = expanded_anno * deletion_masks * expanded_pos_mask
        
        # --- Strategy 2: Only add false positive labels ---
        # Fixed false positive rate
        fixed_fp_ratio = 0.005  # Fixed 0.5% false positive rate
        fp_masks = (rand_tensor < fixed_fp_ratio).float() * expanded_neg_mask
        strategy2_out = expanded_anno + fp_masks
        
        # --- Strategy 3: Delete real labels and add false labels ---
        # Fixed true label retention rate
        fixed_del_ratio_s3 = 0.8  # Fixed 80% retention rate
        rand_tensor_s3 = torch.rand(n_augments, batch_size, n_anno, device=device)
        del_masks_s3 = (rand_tensor_s3 < fixed_del_ratio_s3).float()
        true_part_s3 = expanded_anno * del_masks_s3 * expanded_pos_mask
        
        # Fixed false positive rate
        fixed_fp_ratio_s3 = 0.002  # Fixed 0.2% false positive rate
        rand_tensor_s3_fp = torch.rand(n_augments, batch_size, n_anno, device=device)
        fp_masks_s3 = (rand_tensor_s3_fp < fixed_fp_ratio_s3).float() * expanded_neg_mask
        
        strategy3_out = true_part_s3 + fp_masks_s3
        
        # Merge all strategy results
        strategy_masks = strategy_masks.view(3, n_augments, 1, 1)
        noisy_annos = (
            strategy_masks[0] * strategy1_out +
            strategy_masks[1] * strategy2_out + 
            strategy_masks[2] * strategy3_out
        )
        
        return noisy_annos
    
    def create_rejected_samples_seq(self, orig_seq, similarity, n_augments, args):
        """
        Select the most similar sequence as a rejected sample through the global similarity matrix in all cards (vectorized implementation)
        Args:
            orig_seq: [batch_size, n_seq] Current card's original sequence
            similarity: [batch_size*world_size, batch_size*world_size] Global similarity matrix in all cards
            n_augments: Number of noise versions created for each sample
            args: Contains rank and world_size
        Returns:
            noisy_seqs: [n_augments, batch_size, n_seq] Noisy sequence
        """
        # Collect all card sequences
        orig_seq_all = all_gather_batch([orig_seq])[0]
        
        batch_size, n_seq = orig_seq.shape
        all_batch_size, _ = orig_seq_all.shape
        device = orig_seq.device
        
        # Create tensor for storing results
        noisy_seqs = torch.zeros((n_augments, batch_size, n_seq), device=device)
        
        # Set diagonal values to negative infinity to ensure no self-selection
        sim_for_selection = similarity.clone()
        mask = torch.eye(all_batch_size, device=device, dtype=torch.bool)
        sim_for_selection.masked_fill_(mask, float('-inf'))
        
        # Calculate start and end indices of current process in global batch
        rank = args.rank
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size
        
        # Only focus on similarity rows of current process samples
        local_sim = sim_for_selection[start_idx:end_idx, :]  # [batch_size, all_batch_size]
        
        # For each sample, find the indices of the n_augments samples with the lowest similarity
        _, top_indices = torch.topk(local_sim, k=n_augments, dim=1)  # [batch_size, n_augments]
        
        # Vectorized selection of selected sequences
        # Flatten top_indices and then reshape
        flat_indices = top_indices.view(-1)  # [batch_size*n_augments]
        
        # Get all selected sequences from global sequence pool at once
        selected_flat = orig_seq_all[flat_indices]  # [batch_size*n_augments, n_seq]
        
        # Reshape to [batch_size, n_augments, n_seq]
        selected_seqs = selected_flat.view(batch_size, n_augments, n_seq)
        
        # Transpose to the desired final shape
        noisy_seqs = selected_seqs.permute(1, 0, 2)  # [n_augments, batch_size, n_seq]
        
        return noisy_seqs

    def create_rejected_samples_anno(self, orig_seq, orig_anno, similarity, n_augments, args):
        """
        Select the most similar sequence as a rejected sample through the global similarity matrix in all cards (vectorized implementation)
        Args:
            orig_seq: [batch_size, n_seq] Current card's original sequence
            orig_anno: [batch_size, n_anno] Current card's original annotation
            similarity: [batch_size*world_size, batch_size*world_size] Global similarity matrix in all cards
            n_augments: Number of noise versions created for each sample
            args: Contains rank and world_size
        Returns:
            noisy_anno: [n_augments, batch_size, n_anno] Noisy annotation
        """
        # Collect all card sequences
        orig_seq_all = all_gather_batch([orig_seq])[0]
        orig_anno_all = all_gather_batch([orig_anno])[0]
        
        batch_size, n_anno = orig_anno.shape
        all_batch_size, _ = orig_anno_all.shape
        device = orig_seq.device
        
        # Create tensor for storing results
        noisy_anno = torch.zeros((n_augments, batch_size, n_anno), device=device)
        
        # Set diagonal values to negative infinity to ensure no self-selection
        sim_for_selection = similarity.clone()
        mask = torch.eye(all_batch_size, device=device, dtype=torch.bool)
        sim_for_selection.masked_fill_(mask, float('-inf'))
        
        # Calculate start and end indices of current process in global batch
        rank = args.rank
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size
        
        # Only focus on similarity rows of current process samples
        local_sim = sim_for_selection[start_idx:end_idx, :]  # [batch_size, all_batch_size]
        
        # For each sample, find the indices of the n_augments samples with the lowest similarity
        _, top_indices = torch.topk(local_sim, k=n_augments, dim=1)  # [batch_size, n_augments]
        
        # Vectorized selection of selected sequences
        # Flatten top_indices and then reshape
        flat_indices = top_indices.view(-1)  # [batch_size*n_augments]
        
        # Get all selected sequences from global sequence pool at once
        selected_flat = orig_anno_all[flat_indices]  # [batch_size*n_augments, n_anno]
        
        # Reshape to [batch_size, n_augments, n_seq]
        selected_anno = selected_flat.view(batch_size, n_augments, n_anno)
        
        # Transpose to the desired final shape
        noisy_anno = selected_anno.permute(1, 0, 2)  # [n_augments, batch_size, n_anno]
        
        return noisy_anno

    def create_rejected_predicted_anno(self, orig_seq, orig_anno, similarity, n_augments, args):
        """
        Use model-predicted annotation as rejected samples through vectorized implementation
        Args:
            orig_seq: [batch_size, n_seq] Current card's original sequence
            orig_anno: [batch_size, n_anno] Current card's original annotation
            similarity: [batch_size*world_size, batch_size*world_size] Global similarity matrix in all cards
            n_augments: Number of noise versions created for each sample
            args: Contains rank and world_size
        Returns:
            noisy_anno: [n_augments, batch_size, n_anno] Noisy annotation
        """
        # Collect all card sequences
        orig_seq_all = all_gather_batch([orig_seq])[0]
        orig_anno_all = all_gather_batch([orig_anno])[0]
        
        batch_size, n_anno = orig_anno.shape
        all_batch_size, _ = orig_anno_all.shape
        device = orig_seq.device
        
        # Create tensor for storing results
        noisy_anno = torch.zeros((n_augments, batch_size, n_anno), device=device)
        
        # Set diagonal values to negative infinity to ensure no self-selection
        sim_for_selection = similarity.clone()
        mask = torch.eye(all_batch_size, device=device, dtype=torch.bool)
        sim_for_selection.masked_fill_(mask, float('-inf'))
        
        # Calculate start and end indices of current process in global batch
        rank = args.rank
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size
        
        # Only focus on similarity rows of current process samples
        local_sim = sim_for_selection[start_idx:end_idx, :]  # [batch_size, all_batch_size]
        
        # For each sample, find the indices of the n_augments samples with the lowest similarity
        _, top_indices = torch.topk(local_sim, k=n_augments, dim=1)  # [batch_size, n_augments]
        
        # Vectorized selection of selected sequences
        # Flatten top_indices and then reshape
        flat_indices = top_indices.view(-1)  # [batch_size*n_augments]
        
        # Get all selected sequences from global sequence pool at once
        selected_flat = orig_anno_all[flat_indices]  # [batch_size*n_augments, n_anno]
        
        # Reshape to [batch_size, n_augments, n_seq]
        selected_anno = selected_flat.view(batch_size, n_augments, n_anno)
        
        # Transpose to the desired final shape
        noisy_anno = selected_anno.permute(1, 0, 2)  # [n_augments, batch_size, n_anno]
        
        return noisy_anno

    def log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probabilities for binary annotation task with added false positive penalty weights
        """
        # Use numerically stable method to calculate sigmoid log probabilities
        pos_logprobs = -F.softplus(-logits)  # log(sigmoid(x))
        neg_logprobs = -F.softplus(logits)   # log(1-sigmoid(x))
        
        # Create false positive penalty mask - model predicts positive (sigmoid(logits)>0.5 or logits>0) but label is 0
        fp_mask = (logits > 0) & (labels == 0)
        
        # Create false negative penalty mask - model predicts negative but label is 1
        fn_mask = (logits <= 0) & (labels == 1)
        
        # Base log probabilities
        log_probs = labels * pos_logprobs + (1 - labels) * neg_logprobs
        
        # Apply greater penalty for false positives (reduce log probability)
        # False positive penalty coefficient is 2.0, meaning false positive penalty is 2x normal
        log_probs = torch.where(fp_mask, log_probs * 2.0, log_probs)
        
        return log_probs
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: [(1+n_augments)*batch_size, sequence_length, n_anno]
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: [(1+n_augments)*batch_size, sequence_length]
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape == labels.shape

        per_token_logps = self.log_probs_from_logits(logits, labels)

        logprobs_sums = (per_token_logps).sum(-1)
        logprobs_means = (per_token_logps).mean(-1)
        return logprobs_sums, logprobs_means

    def _get_batch_logps_seq(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: [(1+n_augments)*batch_size, sequence_length, n_token]
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: [(1+n_augments)*batch_size, sequence_length]
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        # per_token_logps = self.log_probs_from_logits(logits, labels)
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=labels.long().unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack(
                [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
            )
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            log_probs_labels = []
            for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
                row_log_probs = F.log_softmax(row_logits, dim=-1)
                row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.long().unsqueeze(-1)).squeeze(-1)
                log_probs_labels.append(row_log_probs_labels)
            log_probs_labels = torch.stack(log_probs_labels)
        logprobs_sums = (log_probs_labels).sum(-1)
        logprobs_means = (log_probs_labels).mean(-1)
        return logprobs_sums, logprobs_means

    def train_step(self, protein_seq, orig_anno=None, args=None):
        """
        DPO training step - use noise data as rejected samples and ignore rejected samples identical to original samples
        """
        # 1. Create rejected data
        rejected_annos = self.create_rejected_samples(orig_anno, n_augments=10)  # [n_augments, batch_size, n_anno]
        n_augments, batch_size, n_anno = rejected_annos.shape
        
        # 2. Check if each rejected sample is identical to original sample
        expanded_orig = orig_anno.unsqueeze(0).expand(n_augments+1, -1, -1)  # [n_augments, batch_size, n_anno]
        is_identical = torch.all(rejected_annos == expanded_orig[1:], dim=2)  # [n_augments, batch_size]
        
        # 3. Create valid sample mask (1=different/valid, 0=same/invalid)
        valid_sample_mask = (~is_identical).float()  # [n_augments, batch_size]
        
        # 4. Record how many invalid samples
        invalid_count = is_identical.sum().item()
        # if invalid_count > 0 and args is not None and args.rank == 0:
        #     print(f"Warning: Detected {invalid_count}/{n_augments*batch_size} rejected samples identical to original samples")
        
        # Continue processing
        combined_anno = torch.cat([orig_anno.unsqueeze(0), rejected_annos], dim=0)  # [1+n_augments, batch_size, n_anno]
        extended_protein_seq = protein_seq.unsqueeze(0).expand(1+n_augments, -1, -1)  # [1+n_augments, batch_size, n_seq]

        # Get sequence features and annotation logits
        _, _, n_seq = extended_protein_seq.shape
        reshaped_protein_seq = extended_protein_seq.reshape(-1, n_seq)
        reshaped_anno = combined_anno.reshape(-1, n_anno)

        # Use model to get logits
        # anno_logits = self.policy_module.get_anno_logits(reshaped_protein_seq)  # [(1+n_augments)*batch_size, n_anno]
        outputs = self.policy_module(protein_seq, orig_anno, "train")
        anno_logits = outputs['anno_logits']
        train_loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs

        # Calculate log probabilities
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            anno_logits.unsqueeze(0).expand(1+n_augments, -1, -1).reshape(-1, n_anno), reshaped_anno, average_log_prob=False
        )

        # Reshape back to [n_augments, batch_size] to separate preferred and rejected samples
        all_logps_sum = all_logps_sum.reshape(1+n_augments, batch_size)
        chosen_logps = all_logps_sum[0]  # [batch_size]
        rejected_logps = all_logps_sum[1:]  # [n_augments, batch_size]

        # Calculate reference model log probabilities (no gradients needed)
        with torch.no_grad():
            ref_anno_logits = self.reference_module.get_anno_logits(protein_seq)
            ref_all_logps_sum, _ = self._get_batch_logps(
                ref_anno_logits.unsqueeze(0).expand(1+n_augments, -1, -1).reshape(-1, n_anno), reshaped_anno, average_log_prob=False
            )
            
            ref_all_logps_sum = ref_all_logps_sum.reshape(1+n_augments, batch_size)
            ref_chosen_logprobs = ref_all_logps_sum[0]  # [batch_size]
            ref_rejected_logprobs = ref_all_logps_sum[1:]  # [n_augments, batch_size]

        # Calculate DPO loss - pass in valid sample mask
        dpo_loss, chosen_rewards, rejected_rewards = self.loss_fn(
            chosen_logps, 
            rejected_logps, 
            ref_chosen_logprobs, 
            ref_rejected_logprobs,
            rejected_weights=valid_sample_mask
        )
        
        # 4. Calculate false positive rewards in rejected samples - but we want rejected samples to have more false positives
        # For rejected samples, more false positives are better because we want the model to learn to avoid this situation
        
        # Modify reward calculation to make false positive and false negative weights different
        # Apply in log_probs_from_logits method
        
        # KL divergence calculation for binary multi-label task
        policy_probs = torch.sigmoid(anno_logits)
        ref_probs = torch.sigmoid(ref_anno_logits)

        # Prevent 0 and 1 to avoid numerical instability
        epsilon = 1e-6
        policy_probs = torch.clamp(policy_probs, epsilon, 1 - epsilon)
        ref_probs = torch.clamp(ref_probs, epsilon, 1 - epsilon)

        # Create Bernoulli distributions using torch.distributions
        policy_dist = Bernoulli(probs=policy_probs)
        ref_dist = Bernoulli(probs=ref_probs)

        # Calculate KL divergence
        kl_penalty = torch.distributions.kl_divergence(ref_dist, policy_dist).mean()
        self.train_weight = 0.01
        self.dpo_weight = 0.01
        self.kl_weight = 1.0
        # Modify total loss
        loss = train_loss * self.train_weight + dpo_loss * self.dpo_weight + kl_penalty * self.kl_weight
        
        # Update model
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        # Update learning rate scheduler
        self.policy_scheduler.step()
        
        # Record metrics including valid sample ratio
        if args.rank == 0 and args.use_wandb:
            with torch.no_grad():
                valid_ratio = valid_sample_mask.mean().item()
                reward_diff = (chosen_rewards - rejected_rewards).mean().item()
                logps_diff = (chosen_logps - rejected_logps.mean(dim=0)).mean().item()
                wandb.log({
                    'dpo/loss': loss.item(),
                    'dpo/dpo_loss': dpo_loss.item(),
                    'dpo/train_loss': train_loss.item(),
                    'dpo/chosen_reward': chosen_rewards.mean().item(),
                    'dpo/rejected_reward': rejected_rewards.mean().item(),
                    'dpo/reward_diff': reward_diff,
                    'dpo/chosen_logps': chosen_logps.mean().item(),
                    'dpo/rejected_logps': rejected_logps.mean().item(),
                    'dpo/logps_diff': logps_diff,
                    'dpo/ref_chosen_logps': ref_chosen_logprobs.mean().item(),
                    'dpo/ref_rejected_logps': ref_rejected_logprobs.mean().item(),
                    'dpo/learning_rate': self.policy_scheduler.get_last_lr()[0],
                    'dpo/kl_penalty': kl_penalty.item(),
                    'dpo/valid_sample_ratio': valid_ratio,  # New: valid sample ratio
                }, commit=True)
        
        return loss

    def train_step_seq(self, protein_seq, orig_anno=None, args=None):
        """
        DPO training step - use noise data as rejected samples
        """
        # Ensure actor model is in training mode
        self.policy_module.train()
        # Ensure reference model is in evaluation mode
        self.reference_module.eval()

        for name, param in self.policy_module.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        #         print(f"Activate LoRA parameters: {name}")
        
        # Existing code...
        outputs = self.policy_module(protein_seq, orig_anno, "train")

        similarity = outputs['similarity'] # [bs*world_size, bs*world_size]
        # 1. Create rejected data
        rejected_seqs = self.create_rejected_samples_seq(protein_seq, similarity, n_augments=3, args=args)  # [n_augments, batch_size, n_seq]
        n_augments, batch_size, n_seq = rejected_seqs.shape

        # Continue processing
        combined_seq = torch.cat([protein_seq.unsqueeze(0), rejected_seqs], dim=0)  # [1+n_augments, batch_size, n_seq]
        extended_protein_seq = protein_seq.unsqueeze(0).expand(1+n_augments, -1, -1)  # [1+n_augments, batch_size, n_seq]

        # Get sequence features and annotation logits
        reshaped_seq = combined_seq.reshape(-1, n_seq)
        # Before calculating reference model output, ensure two models are different instances
        # print("Are Policy and Reference the same instance:", id(self.policy_module) == id(self.reference_module))
        
        # Use model to get logits
        seq_logits = outputs['seq_logits'] # [batch_size, n_seq, n_token]
        
        # Calculate reference model log probabilities (no gradients needed)
        with torch.no_grad():
            ref_outputs = self.reference_module(protein_seq, orig_anno, "test")  # Explicitly use eval mode
            ref_seq_logits = ref_outputs['seq_logits']
        

        train_loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
        bs, n_seq, n_token = seq_logits.shape
        # Calculate log probabilities
        all_logps_sum, all_logps_mean = self._get_batch_logps_seq(
            seq_logits.unsqueeze(0).expand(1+n_augments, -1, -1, -1).reshape(-1, n_seq, n_token), reshaped_seq, average_log_prob=False
        )
        self.nll_loss = -all_logps_mean.mean()
        # Reshape back to [n_augments, batch_size] to separate preferred and rejected samples
        all_logps_sum = all_logps_sum.reshape(1+n_augments, batch_size)
        chosen_logps = all_logps_sum[0]  # [batch_size, n_seq]
        rejected_logps = all_logps_sum[1:]  # [n_augments, batch_size, n_seq]

        # Calculate reference model log probabilities (no gradients needed)
        with torch.no_grad():
            ref_outputs = self.reference_module(protein_seq, orig_anno, "test")
            ref_seq_logits = ref_outputs['seq_logits']
            ref_all_logps_sum, _ = self._get_batch_logps_seq(
                ref_seq_logits.unsqueeze(0).expand(1+n_augments, -1, -1, -1).reshape(-1, n_seq, n_token), reshaped_seq, average_log_prob=False
            )
            
            ref_all_logps_sum = ref_all_logps_sum.reshape(1+n_augments, batch_size)
            ref_chosen_logprobs = ref_all_logps_sum[0]  # [batch_size, n_seq]
            ref_rejected_logprobs = ref_all_logps_sum[1:]  # [n_augments, batch_size, n_seq]
        # print("seq_logits-ref_seq_logits: ", (seq_logits-ref_seq_logits).mean().item())
        # sys.exit()
        # Calculate DPO loss - pass in valid sample mask
        dpo_loss, chosen_rewards, rejected_rewards = self.loss_fn(
            chosen_logps, 
            rejected_logps, 
            ref_chosen_logprobs, 
            ref_rejected_logprobs,
        )
        
        # 4. Calculate false positive rewards in rejected samples - but we want rejected samples to have more false positives
        # For rejected samples, more false positives are better because we want the model to learn to avoid this situation
        
        # Modify reward calculation to make false positive and false negative weights different
        # Apply in log_probs_from_logits method
        
        # Calculate KL divergence for sequence prediction task
        # seq_logits and ref_seq_logits shape should be [batch_size, seq_length, vocab_size]

        # Apply log_softmax to get log probabilities
        log_policy_probs = F.log_softmax(seq_logits, dim=-1)
        log_ref_probs = F.log_softmax(ref_seq_logits, dim=-1)
        ref_probs = F.softmax(ref_seq_logits, dim=-1)

        # Direct KL divergence calculation: KL(p||q) = sum(p * (log(p) - log(q)))
        # This is the mathematical expansion treating each position as one distribution
        kl_penalty = (ref_probs * (log_ref_probs - log_policy_probs)).sum(dim=-1)

        # Take mean over sequence positions and batches
        kl_penalty = kl_penalty.mean()
        self.train_weight = 0.01
        self.dpo_weight = 1.0
        self.kl_weight = 0.1
        self.nll_weight = 0.01
        # Modify total loss
        loss = dpo_loss * self.dpo_weight + self.nll_loss * self.nll_weight + train_loss * self.train_weight
        
        # Update model
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        # Update learning rate scheduler
        self.policy_scheduler.step()
        
        # Record metrics including valid sample ratio
        if args.rank == 0 and args.use_wandb:
            with torch.no_grad():
                # valid_ratio = valid_sample_mask.mean().item()
                reward_diff = (chosen_rewards - rejected_rewards).mean().item()
                logps_diff = (chosen_logps - rejected_logps.mean(dim=0)).mean().item()
                wandb.log({
                    'dpo/loss': loss.item(),
                    'dpo/nll_loss': self.nll_loss.item(),
                    'dpo/dpo_loss': dpo_loss.item(),
                    'dpo/train_loss': train_loss.item(),
                    'dpo/chosen_reward': chosen_rewards.mean().item(),
                    'dpo/rejected_reward': rejected_rewards.mean().item(),
                    'dpo/reward_diff': reward_diff,
                    'dpo/chosen_logps': chosen_logps.mean().item(),
                    'dpo/rejected_logps': rejected_logps.mean().item(),
                    'dpo/logps_diff': logps_diff,
                    'dpo/ref_chosen_logps': ref_chosen_logprobs.mean().item(),
                    'dpo/ref_rejected_logps': ref_rejected_logprobs.mean().item(),
                    'dpo/learning_rate': self.policy_scheduler.get_last_lr()[0],
                    'dpo/kl_penalty': kl_penalty.item(),
                    # 'dpo/valid_sample_ratio': valid_ratio,  # New: valid sample ratio
                }, commit=True)
        
        return loss

    def train_step_anno(self, protein_seq, orig_anno=None, args=None):
        """
        DPO training step - use noise data as rejected samples
        """
        # Ensure actor model is in training mode
        self.policy_module.train()
        # Ensure reference model is in evaluation mode
        self.reference_module.eval()

        for name, param in self.policy_module.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        outputs = self.policy_module(protein_seq, orig_anno, "train")

        similarity = outputs['similarity'] # [bs*world_size, bs*world_size]
        # 1. Create rejected data
        if self.reject_mode == "predict": # Use predicted anno as rejected samples
            rejected_anno = outputs['pred_anno'].unsqueeze(0)
        else: # Use most similar anno in batch as rejected samples
            rejected_anno = self.create_rejected_samples_anno(protein_seq, orig_anno, similarity, n_augments=3, args=args)  # [n_augments, batch_size, n_anno]
        n_augments, batch_size, n_anno = rejected_anno.shape

        # Continue processing
        combined_anno = torch.cat([orig_anno.unsqueeze(0), rejected_anno], dim=0)  # [1+n_augments, batch_size, n_anno]
        extended_orig_anno = orig_anno.unsqueeze(0).expand(1+n_augments, -1, -1)  # [1+n_augments, batch_size, n_anno]

        # Get sequence features and annotation logits
        reshaped_anno = combined_anno.reshape(-1, n_anno)
        # Before calculating reference model output, ensure two models are different instances
        # print("Are Policy and Reference the same instance:", id(self.policy_module) == id(self.reference_module))
        
        # Use model to get logits
        anno_logits = outputs['anno_logits'] # [batch_size, n_anno]
        
        # Calculate reference model log probabilities (no gradients needed)
        with torch.no_grad():
            ref_outputs = self.reference_module(protein_seq, orig_anno, "test")  # Explicitly use eval mode
            ref_anno_logits = ref_outputs['anno_logits']

        train_loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
        itc_loss = outputs['itc_loss'] if isinstance(outputs, dict) else outputs
        diversity_loss = outputs['diversity_loss'] if isinstance(outputs, dict) else outputs
        bs, n_anno = anno_logits.shape
        # Calculate log probabilities
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            anno_logits.unsqueeze(0).expand(1+n_augments, -1, -1).reshape(-1, n_anno), reshaped_anno, average_log_prob=False
        )
        self.nll_loss = -all_logps_mean.mean()
        # Reshape back to [n_augments, batch_size] to separate preferred and rejected samples
        all_logps_sum = all_logps_sum.reshape(1+n_augments, batch_size)
        chosen_logps = all_logps_sum[0]  # [batch_size, n_anno]
        rejected_logps = all_logps_sum[1:]  # [n_augments, batch_size, n_anno]

        # Calculate reference model log probabilities (no gradients needed)
        with torch.no_grad():
            ref_outputs = self.reference_module(protein_seq, orig_anno, "test")
            ref_anno_logits = ref_outputs['anno_logits']
            ref_all_logps_sum, _ = self._get_batch_logps(
                ref_anno_logits.unsqueeze(0).expand(1+n_augments, -1, -1).reshape(-1, n_anno), reshaped_anno, average_log_prob=False
            )
            
            ref_all_logps_sum = ref_all_logps_sum.reshape(1+n_augments, batch_size)
            ref_chosen_logprobs = ref_all_logps_sum[0]  # [batch_size, n_anno]
            ref_rejected_logprobs = ref_all_logps_sum[1:]  # [n_augments, batch_size, n_anno]

        # Calculate DPO loss - pass in valid sample mask
        dpo_loss, chosen_rewards, rejected_rewards = self.loss_fn(
            chosen_logps, 
            rejected_logps, 
            ref_chosen_logprobs, 
            ref_rejected_logprobs,
        )
        
        self.train_weight = 1
        self.dpo_weight = 1.0
        self.nll_weight = 100
        self.diversity_weight = 100
        self.alpha = 1
        self.itc_weight = 1
        # Modify total loss
        loss = (1-self.alpha) * train_loss + self.alpha * dpo_loss + self.nll_loss * self.nll_weight + diversity_loss * self.diversity_weight + itc_loss * self.itc_weight
        
        # Update model
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        # Update learning rate scheduler
        self.policy_scheduler.step()
        
        # Record metrics including valid sample ratio
        if args.rank == 0 and args.use_wandb:
            with torch.no_grad():
                # valid_ratio = valid_sample_mask.mean().item()
                reward_diff = (chosen_rewards - rejected_rewards).mean().item()
                logps_diff = (chosen_logps - rejected_logps.mean(dim=0)).mean().item()
                wandb.log({
                    'dpo/loss': loss.item(),
                    'dpo/itc_loss': itc_loss.item(),
                    'dpo/diversity_loss': diversity_loss.item(),
                    'dpo/nll_loss': self.nll_loss.item(),
                    'dpo/dpo_loss': dpo_loss.item(),
                    'dpo/train_loss': train_loss.item(),
                    'dpo/chosen_reward': chosen_rewards.mean().item(),
                    'dpo/rejected_reward': rejected_rewards.mean().item(),
                    'dpo/reward_diff': reward_diff,
                    'dpo/chosen_logps': chosen_logps.mean().item(),
                    'dpo/rejected_logps': rejected_logps.mean().item(),
                    'dpo/logps_diff': logps_diff,
                    'dpo/ref_chosen_logps': ref_chosen_logprobs.mean().item(),
                    'dpo/ref_rejected_logps': ref_rejected_logprobs.mean().item(),
                    'dpo/learning_rate': self.policy_scheduler.get_last_lr()[0],
                }, commit=True)

        return

    def eval(self, dataloader, args, epoch=0):
        self.eval_count += 1
        
        # Define base perspective
        views = ['cls']
        k_values = [1, 3, 5, 10]
        
        # Initialize metrics dictionary
        metrics = {
            'TEST_ITA_loss': 0.0,
            'TEST_pred_loss': 0.0,
            'TEST_diversity_loss': 0.0,
            'TEST_seq_diversity_loss': 0.0,
            'TEST_anno_diversity_loss': 0.0,
            'TEST_orthogonality_loss': 0.0,
            'TEST_total_loss': 0.0,
            'TEST_anno_accuracy': 0.0,
            'TEST_anno_precision': 0.0,
            'TEST_anno_recall': 0.0,
            'TEST_anno_f1': 0.0,
            'TEST_anno_auroc': 0.0,
            'TEST_anno_fmax': 0.0,
            'TEST_anno_auprc': 0.0,
            'TEST_total_batches': 0
        }
        
        # Add accuracy metrics for all perspectives
        for view in views:
            # Add base accuracy
            metrics[f'TEST_itc_accuracy_{view}'] = 0.0
            metrics[f'TEST_alignment_score_{view}'] = 0.0
            # Add top-k accuracy
            for k in k_values:
                metrics[f'TEST_itc_accuracy_{view}_top{k}'] = 0.0
        
        # Add overall accuracy
        metrics['TEST_itc_accuracy_total'] = 0.0
        for k in k_values:
            metrics[f'TEST_itc_accuracy_total_top{k}'] = 0.0

        # Define metrics update helper function
        def update_metrics_from_outputs(outputs):
            if isinstance(outputs, dict):
                metrics['TEST_ITA_loss'] += outputs['itc_loss'].item()
                metrics['TEST_pred_loss'] += outputs['pred_loss'].item()
                metrics['TEST_total_loss'] += outputs['total_loss'].item()
                
                # Update other metrics
                for k, v in outputs.items():
                    # Add collection for anno prediction metrics
                    if k.startswith('anno_') or k.startswith('itc_accuracy_') or k.startswith('alignment_score_'):
                        metrics_key = f'TEST_{k}'
                        if metrics_key not in metrics:
                            continue
                        metrics[metrics_key] += v.item()
            else:
                metrics['TEST_total_loss'] += outputs.item()
            
            metrics['TEST_total_batches'] += 1
        
        self.policy_module.eval()
        
        with torch.no_grad():
            # Add tqdm progress bar, display only in main process
            dataloader_wrapper = tqdm(
                dataloader,
                desc=f'Evaluation #{self.eval_count}',
                dynamic_ncols=True,
                disable=args.rank != 0  # Non-main process disable progress bar
            )
            for i, current_batch in enumerate(dataloader_wrapper):
                # Process current batch
                input_seq, input_anno = current_batch
                input_seq = input_seq.contiguous().to(args.gpu).squeeze(1)
                input_anno = input_anno.contiguous().float().to(args.gpu)
                
                # Use current batch for processing
                outputs = self.policy_module(input_seq, input_anno, "test", epoch)
                # Use helper function to update metrics
                update_metrics_from_outputs(outputs)
                
                
                # Update progress bar information
                if args.rank == 0:
                    dataloader_wrapper.set_postfix({
                        'loss': f"{metrics['TEST_total_loss']/metrics['TEST_total_batches']:.4f}",
                        'pred_loss': f"{metrics['TEST_pred_loss']/metrics['TEST_total_batches']:.4f}",
                        'itc_loss': f"{metrics['TEST_ITA_loss']/metrics['TEST_total_batches']:.4f}",
                    })

                if i % 10 == 0:  # Perform lightweight cleanup every 10 batches
                    torch.cuda.empty_cache()

        # Simplified version of metrics synchronization - only execute in distributed environment
        if args.world_size > 1:  # Only synchronize in multi-process environment
            for key in metrics:
                tensor = torch.tensor(metrics[key]).to(args.gpu)
                dist.all_reduce(tensor)
                metrics[key] = tensor.item()

        # Calculate average values
        averages = {key: (value / metrics['TEST_total_batches'] if key != 'TEST_total_batches' else value) 
                    for key, value in metrics.items()}
        self.policy_module.train()
        # Cleanup after evaluation
        dataloader_wrapper.close()

        torch.cuda.empty_cache()
        gc.collect()
        return averages
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_lr, init_lr, min_lr, current_step):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_lr = warmup_lr
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.current_step = current_step
        
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.warmup_lr + (self.init_lr - self.warmup_lr) * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def clean_with_dpo(train_loader, test_dataloader, args, actor_model, reference_model, optimizer_actor):
    """Main cleaning function - DPO version"""
    cleaner = DPOTrainer(
        actor_model,
        reference_model,
        beta=args.dpo_beta
    )
    
    # Calculate total training steps and warmup steps
    max_batches = len(train_loader)
    total_steps = max_batches * args.dpo_total_epochs
    warmup_steps = int(0.01 * total_steps)  # 1% steps for warmup
    
    # Optimizer device migration handling
    def move_optimizer_to_device(optimizer, device):
        """Move optimizer state to specified device"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                # Ensure parameters are on correct device
                if param.device != device:
                    print(f"Warning: Parameter not on expected device - from {param.device} to {device}")
                    # No need to move here as parameters should already be in model and moved to correct device
        
        # Move all tensors in optimizer state dict
        for param, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.device != device:
                        # print(f"Moving optimizer state from {v.device} to {device}")
                        state[k] = v.to(device)
    
    # Determine current device
    current_device = next(actor_model.parameters()).device
    print(f"Current model device: {current_device}")
    
    # Move optimizer state to current device
    move_optimizer_to_device(optimizer_actor, current_device)
    
    # Safely set optimizer
    try:
        cleaner.policy_optimizer = optimizer_actor
        print("Successfully set optimizer")
    except Exception as e:
        print(f"Error setting optimizer: {e}")
        # Fallback: create new optimizer
        print("Creating new optimizer...")
        # policy_module = cleaner.policy.module if hasattr(cleaner.policy, 'module') else cleaner.policy
        cleaner.policy_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, cleaner.policy_module.parameters()),
            lr=args.init_lr,
            weight_decay=0.05,
            # betas=(0.9, 0.999)
        )
    
    # Create learning rate scheduler with warmup
    cleaner.policy_scheduler = WarmupCosineScheduler(
        optimizer=cleaner.policy_optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        warmup_lr=args.warmup_lr,
        init_lr=args.init_lr,
        min_lr=args.min_lr,
        current_step=args.actual_epoch*len(train_loader)
    )
    
    best_metric = 0.0  # For saving best model
    
    for epoch in range(args.max_epochs):
        
        # Create progress bar, display only in main process
        train_loader_wrapped = tqdm(
            train_loader,
            desc=f'DPO Epoch {epoch}/{args.max_epochs-1}',
            disable=args.rank != 0,
            dynamic_ncols=True
        )
        
        # Calculate evaluation interval
        # eval_interval = max(1, int(len(train_loader) * 0.1))
        print(f"Epoch {epoch}: Performing evaluation...")
                
        current_metrics = cleaner.eval(test_dataloader, args, epoch)
        
        # Record evaluation metrics
        if args.rank == 0 and args.use_wandb:
            # Record all metrics
            wandb.log({
                **{f'eval/{k}': v for k, v in current_metrics.items()},
            })

        # Train one epoch
        for batch_idx, batch in enumerate(train_loader_wrapped):
                
            # Unpack data
            protein_seq, orig_anno = batch
            
            # Move data to device
            protein_seq = protein_seq.to(args.gpu)
            orig_anno = orig_anno.to(args.gpu)

            # Use DPO training step
            # cleaner.train_step_seq(protein_seq, orig_anno, args)
            cleaner.train_step_anno(protein_seq, orig_anno, args)
            
            # Only update progress bar and record metrics in main process
            if args.rank == 0:
                train_loader_wrapped.set_postfix({
                    'batch': f"{batch_idx}/{max_batches}"  # Add batch progress display
                })

        # Cleanup at end of epoch
        train_loader_wrapped.close()  # Close references that tqdm might hold

        torch.cuda.empty_cache()
        gc.collect()
    print(f"final epoch: Performing evaluation...")
                
    # dist.barrier()  # Ensure all processes synchronize
    current_metrics = cleaner.eval(test_dataloader, args, args.actual_epoch)
    
    # Record evaluation metrics
    if args.rank == 0 and args.use_wandb:
        # Record all metrics
        wandb.log({
            **{f'eval/{k}': v for k, v in current_metrics.items()},
        })

    # Final cleanup after training
    torch.cuda.empty_cache()
    gc.collect()
    
    return


# e(min): 2.32; 2.02; 2.06; 2.12
# t(h): 1:38:54; 2:43:19; 3:53:47; ~5