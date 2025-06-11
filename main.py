import esm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
import argparse
import os
import pandas as pd
import random
import time
import datetime
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
import h5py
import gc
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall, BinaryPrecision, AUROC
from models.protein_clean_v1 import protein_clean_seq_anno_esmc, protein_clean_seq_anno_esmc_zero_shot
from models.cleaning_v4_DPO import clean_with_dpo
from models.cleaning_v3_RW import EnhancedRewardModel, RewardModelTrainer, reward_model_train, CriticFromReward
from evaluation.go_clustering import seq_embedding_analysis_for_go_category, seq_embedding_analysis_for_go_sub_category, seq_embedding_analysis_for_go_sub_category_in_caption_set
from evaluation.go_eval_classification import eval_go_classification
import utils
from utils import warmup_lr_schedule, step_lr_schedule, warmup_lr_schedule_k, step_lr_schedule_k, get_lr_schedule
import subprocess
from itertools import chain
import wandb
import re
import torch.multiprocessing as mp
from copy import deepcopy
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from torch.utils.data import IterableDataset, get_worker_info
import csv
import matplotlib.pyplot as plt
from dataloader import create_loader, create_sampler, Dataset_swissprot
from torch.cuda.amp import autocast
import numpy as np
import torch._dynamo as _dynamo
import shutil
from torch import nn

_dynamo.config.suppress_errors = True

# Create a simple config class to simulate the config of HuggingFace model
@dataclass
class ESMCConfig:
    use_return_dict: bool = True
    is_encoder_decoder: bool = False
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = False
    
    # Add get method to be compatible with PEFT library
    def get(self, key, default=None):
        """Simulate the get method of HuggingFace config class"""
        return getattr(self, key, default)
    
    # Add to_dict method, some PEFT implementations may call it
    def to_dict(self):
        """Return the dictionary representation of the config"""
        return {
            "use_return_dict": self.use_return_dict,
            "is_encoder_decoder": self.is_encoder_decoder,
            "is_decoder": self.is_decoder,
            "add_cross_attention": self.add_cross_attention,
            "tie_word_embeddings": self.tie_word_embeddings
        }

def collate_fn(batch):
    max_length = max(len(s) for s in batch)
    padded_batch = []
    for s in batch:
        padded_s = [ord(ch) for ch in s] + [0] * (max_length - len(s))
        padded_batch.append(padded_s)
    return torch.tensor(padded_batch)


def exists(val):
    return val is not None

def train(model, optimizer, epoch, dataloader, args, test_dataloader):

    model.train()
    torch.cuda.empty_cache()
    
    # Initialize time statistics
    if not hasattr(train, 'total_train_time'):
        train.total_train_time = 0
    if not hasattr(train, 'total_eval_time'):
        train.total_eval_time = 0
    
    # Initialize training metrics statistics
    train_loss = 0.0
    num_batches = 0
    
    batch_start_time = time.time()
    
    # Add tqdm progress bar, only show in main process
    dataloader_wrapper = tqdm(
        dataloader, 
        desc=f'Training Epoch {epoch}', 
        dynamic_ncols=True,
        disable=args.rank != 0  # Disable progress bar in non-main processes
    )
    
    for i, batch in enumerate(dataloader_wrapper):
        try:
            # Use torch.cuda.amp for self-mixed precision training
            with torch.cuda.amp.autocast():
                input_seq, input_anno = batch
                input_seq = input_seq.cuda(non_blocking=True)
                input_anno = input_anno.float().cuda(non_blocking=True)
                
                outputs = model(input_seq, input_anno, "train", epoch)
                loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # Adjust learning rate after each optimizer update
                    if hasattr(train, 'scheduler'):
                        train.scheduler.step()
            
            # Synchronize devices
            torch.cuda.synchronize()
            
            # Update training metrics
            train_loss += loss.item()
            num_batches += 1
            current_avg_loss = train_loss / num_batches
            
            # Record to wandb
            if args.rank == 0 and args.use_wandb:
                # Get the current actual learning rate (considering batch size scaling)
                current_lr = optimizer.param_groups[0]["lr"]
                
                log_dict = {
                    'instant_train_loss': loss.item(),
                    'average_train_loss': current_avg_loss,
                    'learning_rate': current_lr,  # Record the actual learning rate
                    'step': i + epoch * len(dataloader),
                    'train_itc_loss': outputs['itc_loss'].item(),
                    'train_pred_loss': outputs['pred_loss'].item(),
                    'train_diversity_loss': outputs['diversity_loss'].item(),
                    'train_seq_diversity_loss': outputs['seq_diversity_loss'].item(),
                    'train_anno_diversity_loss': outputs['anno_diversity_loss'].item(),
                    'train_orthogonality_loss': outputs['orthogonality_loss'].item(),
                    'train_intra_sac_loss': outputs['intra_sac_loss'].item(),
                    # 'train_consistency_loss': outputs['consistency_loss'].item()
                }
                wandb.log(log_dict)
            
            # Only update progress bar information in main process
            if args.rank == 0:
                dataloader_wrapper.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{current_avg_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',  # Show the actual learning rate
                    'train_time': f'{train.total_train_time:.1f}s'
                })
            
            # Manually clean up unnecessary tensors
            del input_seq, input_anno, outputs, loss
            if i % 10 == 0:  # Clean up every 10 batches
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error in training step {i}: {e}")
            torch.cuda.empty_cache()
            continue
    
    # Clean up at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()  # Explicitly call garbage collection
    
    # Perform final evaluation at the end of the epoch
    batch_time = time.time() - batch_start_time
    train.total_train_time += batch_time
    
    if args.rank == 0:  # Only print in main process
        print(f"\nEpoch completed in {batch_time:.2f}s")
        print(f"Final average training loss: {train_loss/num_batches:.4f}")
        print("Starting evaluation...")
    
    dist.barrier()  # Ensure all processes start evaluation
    eval_start = time.time()
    
    # Clean up memory before evaluation
    torch.cuda.empty_cache()
    gc.collect()
    
    average = eval(model, test_dataloader, args, epoch)
    
    # Clean up memory after evaluation
    torch.cuda.empty_cache()
    gc.collect()
    
    eval_time = time.time() - eval_start
    train.total_eval_time += eval_time
    
    if args.rank == 0:  # Only print in main process
        print(f"Evaluation took {eval_time:.2f}s")
    
    if args.rank == 0 and args.use_wandb:
        wandb.log({
            **average,
            'epoch_final_train_loss': train_loss/num_batches,
            'epoch_total_train_time': train.total_train_time,
            'epoch_total_eval_time': train.total_eval_time,
            'epoch': epoch
        })
    
    dist.barrier()  # Ensure all processes complete the entire epoch
    
    return average


def eval(model, dataloader, args, epoch=0):
    # Use global variable to track eval count
    if not hasattr(eval, 'eval_count'):
        eval.eval_count = 0
    eval.eval_count += 1
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Define basic views
    views = ['cls', 'mean', 'max']
    k_values = [1, 3, 5, 10]
    
    # Initialize metrics dictionary
    metrics = {
        'TEST_ITA_loss': 0.0,
        'TEST_pred_loss': 0.0,
        # 'TEST_consistency_loss': 0.0,
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
    
    # Add accuracy metrics for all views
    for view in views:
        # Add basic accuracy
        metrics[f'TEST_itc_accuracy_{view}'] = 0.0
        metrics[f'TEST_alignment_score_{view}'] = 0.0
        # Add top-k accuracy
        for k in k_values:
            metrics[f'TEST_itc_accuracy_{view}_top{k}'] = 0.0
    
    # Add overall accuracy
    metrics['TEST_itc_accuracy_total'] = 0.0
    for k in k_values:
        metrics[f'TEST_itc_accuracy_total_top{k}'] = 0.0

    model.eval()
    
    # Add tqdm progress bar, only show in main process
    dataloader_wrapper = tqdm(
        dataloader,
        desc=f'Evaluation #{eval.eval_count}',
        dynamic_ncols=True,
        disable=args.rank != 0  # Disable progress bar in non-main processes
    )
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader_wrapper):
            input_seq, input_anno = batch
            input_seq = input_seq.contiguous().to(args.gpu).squeeze(1)
            input_anno = input_anno.contiguous().float().to(args.gpu)
            
            # Use autocast to reduce memory usage
            with torch.cuda.amp.autocast():
                outputs = model(input_seq, input_anno, "test", epoch)
            
            if isinstance(outputs, dict):
                metrics['TEST_ITA_loss'] += outputs['itc_loss'].item()
                metrics['TEST_pred_loss'] += outputs['pred_loss'].item()
                metrics['TEST_total_loss'] += outputs['total_loss'].item()
                
                # Update other metrics
                for k, v in outputs.items():
                    # Add collection of anno prediction metrics
                    if k.startswith('anno_') or k.startswith('itc_accuracy_') or k.startswith('alignment_score_'):
                        metrics_key = f'TEST_{k}'
                        if metrics_key not in metrics:
                            if args.rank == 0:
                                print(f"Warning: unexpected metrics key {metrics_key}")
                            continue
                        metrics[metrics_key] += v.item()
            else:
                metrics['TEST_total_loss'] += outputs.item()
            
            metrics['TEST_total_batches'] += 1
            
            # Update progress bar information
            if args.rank == 0:
                dataloader_wrapper.set_postfix({
                    'loss': f"{metrics['TEST_total_loss']/metrics['TEST_total_batches']:.4f}",
                    'pred_loss': f"{metrics['TEST_pred_loss']/metrics['TEST_total_batches']:.4f}",
                    'itc_loss': f"{metrics['TEST_ITA_loss']/metrics['TEST_total_batches']:.4f}",
                })

            # Manually delete unnecessary variables
            del input_seq, input_anno, outputs
            # Avoid frequent empty_cache calls in the loop to avoid performance loss
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # Clean up memory after evaluation
    torch.cuda.empty_cache()
    gc.collect()

    # Synchronize metrics across all processes
    for key in metrics:
        if key != 'TEST_total_batches':
            tensor = torch.tensor(metrics[key]).to(args.gpu)
            dist.all_reduce(tensor)
            metrics[key] = tensor.item()
        else:
            tensor = torch.tensor(metrics[key]).to(args.gpu)
            dist.all_reduce(tensor)
            metrics[key] = tensor.item()

    # Calculate averages
    averages = {key: (value / metrics['TEST_total_batches'] if key != 'TEST_total_batches' else value) 
                for key, value in metrics.items()}
    
    return averages


def seed_worker(worker_id):
    """
    Set random seed for each worker of DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_seed(seed):
    """
    Set various random seeds
    """
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # CUDA-related settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32
    torch.backends.cudnn.allow_tf32 = False  # Disable TF32
    
    # Set Python's hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed_worker

# Improved adapter class - preserve all key methods of the original ESMC
class ESMCAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = ESMCConfig()
        
        # Save key attributes of the original model
        self.embedding_dim = model.embed.embedding_dim
        self.embed = model.embed  # Directly save embed attribute
        self._tokenize = model._tokenize
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Process input_ids as positional argument
        if input_ids is not None:
            return self.model(input_ids)
        elif len(kwargs) > 0:
            return self.model(**kwargs)
        else:
            raise ValueError("No valid parameters received")

def main(args):
    
    mp.set_start_method('spawn', force=True)
    args.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # Based on version settings, set the switch
    if args.version == "proclean_itc_alm":
        args.use_itc = True
        args.use_nitc = False
        args.use_alm = True
    elif args.version == "proclean_itc":
        args.use_itc = True
        args.use_nitc = False
        args.use_alm = False
    elif args.version == "proclean_alm":
        args.use_itc = False
        args.use_nitc = False
        args.use_alm = True
    elif args.version == "proclean_nitc_alm":
        args.use_itc = False
        args.use_nitc = True
        args.use_alm = True
    else:
        print(f"Warning: Unknown version {args.version}, using default settings")

    if args.distributed:
        utils.init_distributed_mode(args)  # Use enhanced init_distributed_mode

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = "cuda:%d" % args.local_rank
    args.world_size = utils.get_world_size()
    args.rank = utils.get_rank()

    # Set random seeds
    seed_worker = setup_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # Adjust effective_batch_size based on actual GPU number
    args.effective_batch_size = args.batch_size * args.world_size
    print(f"Using {args.world_size} GPUs")
    print(f"Per GPU batch size: {args.batch_size}")
    print(f"Effective batch size: {args.effective_batch_size}")
    
    # Automatically calculate gradient accumulation steps based on total batch size
    if args.gradient_accumulation_steps <= 0:
        target_batch_size = args.target_batch_size  # New parameter: target batch size
        args.gradient_accumulation_steps = max(1, target_batch_size // args.effective_batch_size)
        print(f"Automatically setting gradient_accumulation_steps to {args.gradient_accumulation_steps}")
        print(f"Final effective batch size with accumulation: {args.effective_batch_size * args.gradient_accumulation_steps}")

    seed = args.seed
    np.random.seed(seed)
    torch.initial_seed()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project_name,
                   name=args.dataset + "_" + args.mode + "_e_" + str(args.actual_epoch) + "_" + args.version + '_bs_%d_dp_%d_h_%d' % (
                            args.batch_size, args.depth, args.attn_heads),
                   entity='AnnoDPO',
                   )

    total_epoch = args.actual_epoch
    
    finetune_checkpoint = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_clean_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP.pth' % (
                                           args.depth, args.attn_heads, 256, 0))
    load_finetune_checkpoint_lora = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_clean_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP_lora.pth' % (
                                           args.depth, args.attn_heads, 256, 0)) if args.actual_epoch == 0 else os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_clean_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP_lora_DPO.pth' % (
                                           args.depth, args.attn_heads, args.batch_size, total_epoch-1))
    finetune_checkpoint_lora = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_clean_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP_lora_DPO.pth' % (
                                           args.depth, args.attn_heads, args.batch_size, total_epoch))
    reward_checkpoint = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_reward_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP.pth' % (
                                           args.depth, args.attn_heads, args.batch_size, total_epoch))
    
    rlhf_checkpoint = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.version}_rlhf_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_DPO.pth' % (
                                           args.depth, args.attn_heads, args.batch_size, total_epoch))
    load_rlhf_checkpoint = os.path.join(args.load_dir,
                                       f'{args.dataset}_{args.version}_rlhf_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_DPO.pth' % (
                                           args.depth, args.attn_heads, args.batch_size, total_epoch-1)) if args.actual_epoch > 0 else None
    load_checkpoint = os.path.join(args.load_dir,
                                       f'{args.dataset}_{args.version}_clean_after_ft_depth_%d_heads_%d_bs_%d_e_%02d_MLP.pth' % (
                                           args.depth, args.attn_heads, 128, total_epoch-1))
    
    if os.path.isfile(finetune_checkpoint) and args.mode == "finetune":
        print(f"{finetune_checkpoint} exist !!!!")
        raise RuntimeError

    # Load ESMC model and wrap it with adapter
    base_model = ESMC.from_pretrained("esmc_300m")
    seq_model = ESMCAdapter(base_model)
    
    # Add a config attribute to the ESMC model
    seq_model.config = ESMCConfig()
    
    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=16,                    # LoRA rank
        lora_alpha=32,           # Scaling factor
        # Correctly specify the linear layer module names
        target_modules=[
            # Linear layers in attention modules
            "layernorm_qkv.1",   # QKV projection
            "out_proj",          # Output projection
            # Feedforward network linear layers
            "ffn.1",             # Feedforward network first linear layer
            "ffn.3",             # Feedforward network second linear layer
            # Sequence head
            "sequence_head.0",   # Sequence head first linear layer
            "sequence_head.3",   # Sequence head second linear layer
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    # Apply LoRA
    seq_model = get_peft_model(seq_model, lora_config)
    
    # Check trainable parameters
    if args.rank == 0:
        total_params = sum(p.numel() for p in seq_model.parameters())
        trainable_params = sum(p.numel() for p in seq_model.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params:,}")
        print(f"Trainable number of parameters: {trainable_params:,}")
        trainable_percent = 100 * trainable_params / total_params
        print(f"Trainable parameter ratio: {trainable_percent:.2f}%")
    
    test_dataset = Dataset_swissprot(mode="test", model=seq_model)
    test_sampler = create_sampler(test_dataset, True, args.world_size, args.rank)
    test_dataloader = \
        create_loader(test_dataset, test_sampler, batch_size=args.test_batch_size, num_workers=8, is_training=False, collate_fn=test_dataset.collate_fn)

    if args.mode == "finetune":
        train_dataset = Dataset_swissprot(mode="train", model=seq_model)
        train_sampler = create_sampler(train_dataset, True, args.world_size, args.rank)
        train_dataloader = \
            create_loader(
                train_dataset, 
                train_sampler, 
                batch_size=args.batch_size,
                num_workers=8,
                is_training=True,
                collate_fn=train_dataset.collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed)
            )
        if args.actual_epoch==0:
            args.checkpoint = None
        else:
            args.checkpoint = load_checkpoint
            args.finetune_epoch = 10
            args.batch_size = 32
        model, model_without_ddp, _, optimizer_finetune = create_model(args,
                                                                                           seq_model,
                                                                                           args.checkpoint, )

        dist.barrier()  # Ensure all processes start training
        model.train()
        
        # Start training loop
        for epoch in range(args.finetune_epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            try:
                finetune_one_epoch(model, optimizer_finetune, train_dataloader, args, epoch, test_dataloader)
            except Exception as e:
                print(f"Training error: {e}")
                if args.distributed:
                    dist.destroy_process_group()
                raise e

        if args.rank == 0:
            print(f'Saving checkpoint for epoch {total_epoch}')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer_finetune': optimizer_finetune.state_dict(),
                'epoch': total_epoch + 1,
            }

            torch.save(save_obj, finetune_checkpoint)
            model_without_ddp.seq_encoder.save_pretrained(finetune_checkpoint_lora)
        print("finetuning finished")

    if args.mode.startswith("caption"):
        save_dir = f"{args.captioned_seq_sav_dir}/epoch{args.actual_epoch}"
        caption_dataset = Dataset_swissprot(mode="train", model=seq_model)
        caption_sampler = create_sampler(caption_dataset, True, args.world_size, args.rank)
        caption_dataloader = \
            create_loader(
                caption_dataset, 
                caption_sampler, 
                batch_size=args.batch_size,
                num_workers=1,
                prefetch_factor=4,
                pin_memory=True,
                persistent_workers=True,
                is_training=True,
                collate_fn=caption_dataset.collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed)
            )
        args.checkpoint = finetune_checkpoint if args.checkpoint is None else args.checkpoint

        if args.mode == "caption_rlhf":
            
            # create eval dataloader
            eval_dataset = Dataset_swissprot(mode="test", model=seq_model)
            eval_sampler = create_sampler(eval_dataset, True, args.world_size, args.rank)
            eval_dataloader = create_loader(
                eval_dataset, 
                eval_sampler, 
                batch_size=args.test_batch_size, 
                num_workers=1, 
                prefetch_factor=4,
                pin_memory=True,
                persistent_workers=True,
                is_training=False, 
                collate_fn=eval_dataset.collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            # get model dict and unpack
            # models = create_dpo_model(args, seq_model, finetune_checkpoint)
            models = create_dpo_model_seq_lora(args, seq_model, finetune_checkpoint, load_finetune_checkpoint_lora, load_rlhf_checkpoint, args.actual_epoch)
            if args.rank == 0 and args.use_wandb:  # only in main process
                wandb.watch(models['actor'], log="parameters", log_freq=100)
            actor_model = models['actor']
            reference_model = models['reference'] 
            optimizer_actor = models['optimizer']
            # pass unpacked model and eval dataloader
            clean_with_dpo(
                train_loader=caption_dataloader,
                test_dataloader=eval_dataloader,
                args=args,
                actor_model=actor_model,
                reference_model=reference_model,
                optimizer_actor=optimizer_actor
            )

            if args.rank == 0:
                print(f'Saving checkpoint for epoch {args.actual_epoch}')
                save_obj = {
                    'actor': actor_model.module.state_dict(),
                    'optimizer_finetune': optimizer_actor.state_dict(),
                    'epoch': args.actual_epoch,
                }
                actor_model.module.seq_encoder.save_pretrained(finetune_checkpoint_lora)
                torch.save(save_obj, rlhf_checkpoint)

            print("rlhf training finished")


    if args.mode.startswith("eval"):
        if args.mode.startswith("eval_freq_"): # eval_freq_high, eval_freq_medium, eval_freq_low
            eval_dataset = Dataset_swissprot(mode=args.mode, model=seq_model)
        elif args.mode.startswith("eval_sub_go_analysis"): # eval_sub_go_analysis_caption_set
            eval_dataset = Dataset_swissprot(mode="caption_rlhf", model=seq_model)
        else:
            eval_dataset = Dataset_swissprot(mode="test", model=seq_model)
        eval_sampler = create_sampler(eval_dataset, True, args.world_size, args.rank)
        eval_dataloader = \
            create_loader(eval_dataset, eval_sampler, batch_size=args.test_batch_size, num_workers=8, is_training=False, collate_fn=eval_dataset.collate_fn)
        
        if args.mode.endswith("rlhf"): # eval_freq_high_rlhf, eval_freq_medium_rlhf, eval_freq_low_rlhf
            models = create_dpo_model_seq_lora(args, seq_model, finetune_checkpoint, load_finetune_checkpoint_lora, load_rlhf_checkpoint, args.actual_epoch)
            if args.rank == 0 and args.use_wandb:  # Only call in main process
                wandb.watch(models['actor'], log="parameters", log_freq=100)
            model = models['actor']
            args.version = args.version + "_rlhf"
        else:
            args.checkpoint = finetune_checkpoint if args.checkpoint is None else args.checkpoint
            model, model_without_ddp, _, optimizer_finetune = create_model(args,
                                                                                            seq_model,
                                                                                            args.checkpoint, )

        
        if args.mode == "eval_go_analysis_rlhf" or args.mode == "eval_go_analysis":
            seq_embedding_analysis_for_go_category(model, args.version)
        elif args.mode == "eval_sub_go_analysis_rlhf":
            seq_embedding_analysis_for_go_sub_category(model, args.version)
        elif args.mode == "eval_sub_go_analysis_caption_set_rlhf":
            seq_embedding_analysis_for_go_sub_category_in_caption_set(model, "dpo", eval_dataloader)
        elif args.mode.startswith("eval_go_classification"):
            eval_go_classification(model, eval_dataloader, args)
        else:
            average = eval(model, eval_dataloader, args)
            print(average)

    # Add zero-shot mode after the condition branch of mode=="eval"
    elif args.mode.startswith("zeroshot"):
        print("Running in zero-shot mode...")
        
        # Create zero-shot model
        zero_shot_model = protein_clean_seq_anno_esmc_zero_shot(seq_model=seq_model, args=args)
    
        # Put the model on the correct device
        zero_shot_model = zero_shot_model.to(args.gpu)
        
        # If using distributed training, wrap the model
        if args.distributed:
            zero_shot_model = torch.nn.parallel.DistributedDataParallel(
                zero_shot_model, 
                device_ids=[args.gpu],
                find_unused_parameters=True
            )
                
        # Load dataset
        if args.mode.startswith("zeroshot_freq_"):
            eval_dataset = Dataset_swissprot(mode=args.mode, model=seq_model)
        else:
            eval_dataset = Dataset_swissprot(mode="test", model=seq_model)
        eval_sampler = create_sampler(eval_dataset, True, args.world_size, args.rank)
        eval_dataloader = create_loader(
            eval_dataset, 
            eval_sampler, 
            batch_size=args.test_batch_size, 
            num_workers=8, 
            is_training=False, 
            collate_fn=eval_dataset.collate_fn
        )
        if args.mode == "zeroshot_go_pred" or args.mode.startswith("zeroshot_freq_"):
            
            # Run evaluation
            print("Starting zero-shot evaluation...")
            metrics = zero_shot_eval(zero_shot_model, eval_dataloader, args)
            
            # Print evaluation results
            if args.rank == 0:
                print("\nZero-shot Evaluation Results:")
                for k, v in metrics.items():
                    if k.startswith('TEST_'):
                        print(f"{k.replace('TEST_', '')}: {v:.4f}")

        elif args.mode == "zeroshot_go_analysis":
            print("Starting GO category embedding analysis...")
            seq_embedding_analysis_for_go_category(zero_shot_model, "zeroshot")
            
        elif args.mode == "zeroshot_go_classification":
            eval_go_classification(zero_shot_model, eval_dataloader, args)

def finetune_one_epoch(model, optimizer_finetune, finetune_dataloader, args, epoch, test_dataloader):
    if args.rank == 0 and args.use_wandb:
        wandb.log({'Epoch': epoch})

    epoch_start = time.time()
    print(f'Finetuning Epoch: {epoch}')
    num_training_steps = len(finetune_dataloader)
    # Calculate actual total steps (num_training_steps is already the number of steps per GPU)
    total_steps = num_training_steps // args.gradient_accumulation_steps  # Consider gradient accumulation
    
    # Calculate warmup steps
    num_warmup_steps = int(args.warmup_epochs * total_steps)

    # Create learning rate scheduler
    if not hasattr(train, 'scheduler'):  # Change to train's attribute
        
        train.scheduler = get_lr_schedule(
            optimizer_finetune, 
            args,
            num_warmup_steps
        )
    
    # Train one epoch
    train(model, optimizer_finetune, epoch, finetune_dataloader, args, test_dataloader)
    
    if args.distributed:
        dist.barrier()

    total_time = time.time() - epoch_start
    print(f'Finetuning time {total_time / 60.0} mins')


def create_model(args, seq_model, checkpoint=None):
    print("Creating model")
    
    # Ensure CUDA device is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Set CUDA device
    torch.cuda.set_device(args.local_rank)
    
    # Create model
    model = protein_clean_seq_anno_esmc(seq_model=seq_model, args=args)
    # Print number of parameters for each part
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = 0
    print("\nModel parameters:")
    print("-" * 50)
    
    # ESM encoder
    esm_params = count_parameters(model.seq_encoder) / 1e6
    total_params += esm_params * 1e6
    print(f"ESM Encoder: {esm_params:.2f}M params")
    
    # Annotation encoder
    anno_encoder_params = count_parameters(model.anno_encoder) / 1e6
    total_params += anno_encoder_params * 1e6
    print(f"Annotation Encoder: {anno_encoder_params:.2f}M params")
    
    # ITC components
    if args.use_itc:
        itc_params = (count_parameters(model.anno_proj) + 
                     count_parameters(model.seq_proj)) / 1e6
        total_params += itc_params * 1e6
        print(f"ITC Components: {itc_params:.2f}M params")
    
    # ALM components
    if args.use_alm:
        alm_params = count_parameters(model.anno_predictor) / 1e6
        total_params += alm_params * 1e6
        print(f"ALM Components: {alm_params:.2f}M params")
    
    print("-" * 50)
    print(f"Total Trainable Parameters: {total_params/1e6:.2f}M")
    print("-" * 50)
    
    optimizer_finetune = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.init_lr,  # Base learning rate, scaling will be handled in scheduler
        weight_decay=0.05  # Add weight decay
    )

    # Load checkpoint
    start_epoch = 0
    if checkpoint is not None:
        print(f"load ckpt: {checkpoint}!")
        checkpoint = torch.load(checkpoint, map_location='cuda:{}'.format(args.local_rank))
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']
        
        # Load model state
        state_dict = on_load_checkpoint(state_dict)
        model.load_state_dict(state_dict, strict=True)
    
    # Ensure all parameters are on the correct CUDA device
    model = model.cuda(args.local_rank)
    
    model_without_ddp = model # if no distributed
    
    # Optimize CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Add memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Distributed training settings
    if args.distributed:
        dist.barrier()
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False  # Reduce communication overhead
        )
        model_without_ddp = model.module
        dist.barrier()

    return model, model_without_ddp, start_epoch, optimizer_finetune


def create_dpo_model(args, seq_model, checkpoint=None):
    print("Creating dpo model")
    
    # Ensure CUDA device is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Set CUDA device
    torch.cuda.set_device(args.local_rank)
    
    # Create model
    actor_model = protein_clean_seq_anno_esmc(seq_model=seq_model, args=args)
    reference_model = protein_clean_seq_anno_esmc(seq_model=seq_model, args=args)
    
    # Load actor checkpoint
    if checkpoint is not None:
        print(f"load ckpt: {checkpoint}!")
        checkpoint = torch.load(checkpoint, map_location='cuda:{}'.format(args.local_rank))
        state_dict = checkpoint['model']
        optimizer_state_dict = checkpoint['optimizer_finetune']
        optimizer_state_dict = on_load_checkpoint(optimizer_state_dict)
        # Load model state
        state_dict = on_load_checkpoint(state_dict)
        actor_model.load_state_dict(state_dict, strict=True)
        reference_model.load_state_dict(state_dict, strict=True)
    
    # Freeze reference model
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    actor_model.seq_encoder.eval()
    for param in actor_model.seq_encoder.parameters():
        param.requires_grad = False
    
    optimizer_actor = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, actor_model.parameters()),
        lr=args.init_lr,
        weight_decay=0.05,
        # betas=(0.9, 0.999)
    )
    optimizer_actor.load_state_dict(optimizer_state_dict)

    # Freeze actor_model at module level selectively
    # First freeze all parameters
    actor_model.eval()
    for param in actor_model.parameters():
        param.requires_grad = False
    
    # Then only unfreeze the key modules related to get_anno_logits
    print("Only keep the gradient of the modules related to get_anno_logits...")
    
    # According to the get_anno_logits function, only the following modules need to keep the gradient:
    # 1. seq_proj - Used for basic projection
    # 2. alignment_layer - Used for feature alignment
    # 3. anno_predictor - Used for generating annotation logits
    # seq_encoder is already frozen using torch.no_grad() in the function

    # Unfreeze these modules
    trainable_params = 0
    for module_name in ['seq_proj', 'alignment_layer', 'anno_predictor']:
        if hasattr(actor_model, module_name):
            module = getattr(actor_model, module_name)
            module.train()
            module_params = sum(p.numel() for p in module.parameters())
            trainable_params += module_params
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfreeze {module_name} module, number of parameters: {module_params:,}")
    
    print(f"Total trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Move model to GPU
    actor_model = actor_model.cuda(args.local_rank)
    reference_model = reference_model.cuda(args.local_rank)
    
    # Distributed training settings
    if args.distributed:
        # Only wrap the model that needs to be trained
        actor_model = torch.nn.parallel.DistributedDataParallel(
            actor_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,  # Since most parameters are frozen, this is necessary
            broadcast_buffers=False
        )
        # Synchronize all processes
        dist.barrier()
    
    # Print parameter statistics (only in main process)
    if args.rank == 0:
        print("\nDPO Model parameters:")
        print("-" * 50)
        
        def print_model_params(model, name):
            # Count trainable and frozen parameters separately
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            
            print(f"{name}:")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
            print(f"  Frozen parameters: {frozen_params/1e6:.2f}M")
            print("-" * 30)
            
            # If it is the actor model, list all trainable modules
            if name == "Actor Model":
                print("  Trainable modules:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"    - {name}: {param.numel()/1e3:.1f}K params")
                print("-" * 30)
        
        print_model_params(actor_model, "Actor Model")
        print_model_params(reference_model, "Reference Model (frozen)")
        print("-" * 50)
    
    return {
        'actor': actor_model,
        'reference': reference_model,
        'optimizer': optimizer_actor
    }

def create_dpo_model_seq_lora(args, seq_model, model_checkpoint, lora_checkpoint=None, load_rlhf_checkpoint=None, epoch=0):
    print("Creating dpo model")
    
    # Ensure CUDA device is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Set CUDA device
    torch.cuda.set_device(args.local_rank)
    
    # Create independent seq_model copies for actor and reference
    from copy import deepcopy
    actor_seq_model = deepcopy(seq_model)
    ref_seq_model = deepcopy(seq_model)
    
    if lora_checkpoint is not None:
        from peft import PeftModel

        # Apply LoRA weights to actor model
        actor_seq_model = PeftModel.from_pretrained(
            actor_seq_model,
            lora_checkpoint,
            is_trainable=True  # actor trainable
        )

        # Apply the same LoRA weights to the reference model, but set it to not trainable
        ref_seq_model = PeftModel.from_pretrained(
            ref_seq_model,
            lora_checkpoint,
            is_trainable=False  # reference not trainable
        )
        print(f"Loaded LoRA weights from {lora_checkpoint}")
        
        # Ensure all parameters of the reference model are frozen
        for param in ref_seq_model.parameters():
            param.requires_grad = False
        for param in actor_seq_model.parameters():
            param.requires_grad = True

    # Create actor and reference models
    actor_model = protein_clean_seq_anno_esmc(seq_model=actor_seq_model, args=args)
    reference_model = protein_clean_seq_anno_esmc(seq_model=ref_seq_model, args=args)
    
    # Load actor checkpoint
    
    print(f"load ckpt: {model_checkpoint}!")
    reference_checkpoint = torch.load(model_checkpoint, map_location='cuda:{}'.format(args.local_rank))
    ref_state_dict = reference_checkpoint['model']
    if epoch == 0:
        actor_checkpoint = model_checkpoint
        actor_checkpoint = torch.load(actor_checkpoint, map_location='cuda:{}'.format(args.local_rank))
        actor_state_dict = actor_checkpoint['model']
    else:
        actor_checkpoint = load_rlhf_checkpoint
        actor_checkpoint = torch.load(actor_checkpoint, map_location='cuda:{}'.format(args.local_rank))
        actor_state_dict = actor_checkpoint['actor']
    actor_state_dict = on_load_checkpoint(actor_state_dict)
    ref_state_dict = on_load_checkpoint(ref_state_dict)
    
    
    # Filter out parameters related to seq_encoder
    actor_dict = actor_model.state_dict()
    ref_dict = reference_model.state_dict()
    
    # Only load parameters related to seq_encoder (avoid conflicts with LoRA)
    actor_filtered_state_dict = {k: v for k, v in actor_state_dict.items() 
                            if k in actor_dict and 'seq_encoder' not in k}
    ref_filtered_state_dict = {k: v for k, v in ref_state_dict.items() 
                            if k in ref_dict and 'seq_encoder' not in k}
    
    # Update model dictionary and load
    actor_dict.update(actor_filtered_state_dict)
    ref_dict.update(ref_filtered_state_dict)
    
    actor_model.load_state_dict(actor_dict, strict=True)
    reference_model.load_state_dict(ref_dict, strict=True)
    
    # Print loading status
    print(f"actor successfully loaded {len(actor_filtered_state_dict)} parameters (excluding seq_encoder parameters)")
    print(f"reference successfully loaded {len(ref_filtered_state_dict)} parameters (excluding seq_encoder parameters)")
    # sys.exit()
    # Process optimizer state
    if 'optimizer_finetune' in actor_checkpoint:
        optimizer_state_dict = actor_checkpoint['optimizer_finetune']
        optimizer_state_dict = on_load_checkpoint(optimizer_state_dict)
        # Use this state for creating optimizer later
    else:
        optimizer_state_dict = None
        print("Optimizer state not found, will reinitialize optimizer")

    
    # Ensure reference model is completely frozen
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
        
    # Create optimizer, only optimize actor model
    optimizer_actor = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, actor_model.parameters()),
        lr=args.init_lr,
        weight_decay=0.05,
        # betas=(0.9, 0.999)
    )
    # Print optimizer_state_dict's key

    # If there is optimizer state, try to load (may need further adaptation)
    if optimizer_state_dict is not None:
        try:
            optimizer_actor.load_state_dict(optimizer_state_dict)
            print("Successfully loaded optimizer state")
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")

    
    # Move model to GPU
    actor_model = actor_model.cuda(args.local_rank)
    reference_model = reference_model.cuda(args.local_rank)
    
    # Distributed training settings
    if args.distributed:
        # Only wrap the model that needs to be trained
        actor_model = torch.nn.parallel.DistributedDataParallel(
            actor_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,  # Since most parameters are frozen, this is necessary
            broadcast_buffers=False
        )
        # Synchronize all processes
        dist.barrier()
    
    # Print parameter statistics (only in main process)
    if args.rank == 0:
        print("\nDPO Model parameters:")
        print("-" * 50)
        
        def print_model_params(model, name):
            # Count trainable and frozen parameters separately
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            
            print(f"{name}:")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
            print(f"  Frozen parameters: {frozen_params/1e6:.2f}M")
            print("-" * 30)
        
        print_model_params(actor_model, "Actor Model")
        print_model_params(reference_model, "Reference Model (frozen)")
        print("-" * 50)
    
    return {
        'actor': actor_model,
        'reference': reference_model,
        'optimizer': optimizer_actor
    }

def on_load_checkpoint(state_dict):
    keys_list = list(state_dict.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            state_dict[deal_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def count_shared_parameters(state_dict, model_dict):
    """
    Calculate the number of shared parameters
    
    Args:
        state_dict: Pre-trained model parameters dictionary
        model_dict: Current model parameters dictionary
    
    Returns:
        total_params: Total shared parameters
        shared_layers: Dictionary of shared layer names and parameter counts
    """
    shared_params = 0
    shared_layers = {}
    
    # Find shared parameters
    for k, v in state_dict.items():
        if k in model_dict:
            params = v.numel()
            shared_params += params
            shared_layers[k] = params
    
    return shared_params, shared_layers

def zero_shot_eval(model, dataloader, args):
    """Function specifically for zero-shot evaluation"""
    # Initialize metrics dictionary
    metrics = {
        'TEST_pred_loss': 0.0,
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
    
    model.eval()
    
    # Add tqdm progress bar, only show in main process
    dataloader_wrapper = tqdm(
        dataloader,
        desc='Zero-shot Evaluation',
        dynamic_ncols=True,
        disable=args.rank != 0  # Disable progress bar in non-main processes
    )
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader_wrapper):
            input_seq, input_anno = batch
            input_seq = input_seq.contiguous().to(args.gpu).squeeze(1)
            input_anno = input_anno.contiguous().float().to(args.gpu)
            
            # Use autocast to reduce memory usage
            with torch.cuda.amp.autocast():
                outputs = model(input_seq, input_anno)
            
            # Update metrics
            metrics['TEST_pred_loss'] += outputs['pred_loss'].item()
            metrics['TEST_total_loss'] += outputs['pred_loss'].item()  # Only prediction loss in zero-shot mode
            
            # Update other metrics
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'fmax', 'auprc']:
                metric_key = f'anno_{metric_name}'
                if metric_key in outputs:
                    metrics[f'TEST_{metric_key}'] += outputs[metric_key].item()
            
            metrics['TEST_total_batches'] += 1
            
            # Update progress bar information
            if args.rank == 0:
                dataloader_wrapper.set_postfix({
                    'loss': f"{metrics['TEST_total_loss']/metrics['TEST_total_batches']:.4f}",
                    'f1': f"{metrics['TEST_anno_f1']/metrics['TEST_total_batches']:.4f}",
                })
            
            # Delete unnecessary variables
            del input_seq, input_anno, outputs
            # Avoid performance loss by not calling empty_cache frequently in the loop
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # Clean up after evaluation
    torch.cuda.empty_cache()
    gc.collect()

    # Synchronize metrics across all processes
    for key in metrics:
        if key != 'TEST_total_batches':
            tensor = torch.tensor(metrics[key]).to(args.gpu)
            dist.all_reduce(tensor)
            metrics[key] = tensor.item()
        else:
            tensor = torch.tensor(metrics[key]).to(args.gpu)
            dist.all_reduce(tensor)
            metrics[key] = tensor.item()
    
    # Calculate average
    num_batches = metrics['TEST_total_batches']
    for key in metrics:
        if key != 'TEST_total_batches':
            metrics[key] /= num_batches
    
    return metrics


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    ## dataset settings
    parser.add_argument('--dataset', default="uniref50", type=str)  # or eval or test or downstream

    ## model settings
    parser.add_argument('--version', default="proclean_itc_alm", type=str,
                        help='[proclean_alm, proclean_itc_alm, proclean_nitc_alm]')
    parser.add_argument('--use_itc', default=True, type=bool)
    parser.add_argument('--use_nitc', default=True, type=bool)
    parser.add_argument('--use_itm', default=True, type=bool)
    parser.add_argument('--use_alm', default=True, type=bool)
    parser.add_argument('--use_slm', default=True, type=bool)
    parser.add_argument('--from_scratch', default=False, type=bool)

    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--anno_lm_loss_weight', default=1., type=float)
    parser.add_argument('--seq_lm_loss_weight', default=1., type=float)
    parser.add_argument('--itc_loss_weight', default=1., type=float)
    parser.add_argument('--itm_loss_weight', default=1., type=float)

    ## structure settings
    parser.add_argument('--seq_len', default=512, type=int)
    parser.add_argument('--num_annotation', default=7533, type=int)
    
    # CrossModalPredictor parameters
    parser.add_argument('--predictor_dropout', default=0.1, type=float,
                        help='Dropout rate in CrossModalPredictor')
    parser.add_argument('--num_residual_blocks', default=2, type=int,
                        help='Number of residual blocks in CrossModalPredictor')
    
    parser.add_argument('--attn_dim_head', default=64, type=int)
    parser.add_argument('--attn_heads', default=8, type=int)  # 35m: 8
    parser.add_argument('--depth', default=12, type=int)  # 35m: 12
    parser.add_argument('--dim', default=960, type=int)
    parser.add_argument('--dim_global', default=512, type=int)
    parser.add_argument('--wide_conv_dilation', default=5, type=int)
    parser.add_argument('--wide_conv_kernel', default=9, type=int)
    parser.add_argument('--glu_conv', default=False, type=bool)
    parser.add_argument('--local_self_attn', default=True, type=bool)
    parser.add_argument('--narrow_conv_kernel', default=9, type=int)
    parser.add_argument('--num_global_tokens', default=1, type=int)
    parser.add_argument('--seq_sample_token_prob', default=0.15, type=float)
    parser.add_argument('--remove_annotation_prob', default=0.25, type=float)
    parser.add_argument('--remove_all_annotations_prob', default=0.3, type=float)  # before: 0.5

    ## training settings
    parser.add_argument('--ft_first', default=True, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--mode', default="clean", type=str)  # or eval or test or downstream
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--actual_epoch', default=0, type=int)
    parser.add_argument('--pretrain_epoch', default=1, type=int)
    parser.add_argument('--fewshot_epoch', default=200, type=int)
    parser.add_argument('--finetune_epoch', default=80, type=int)
    parser.add_argument('--batch_size', default=48, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--caption_batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--sample_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)

    ### dir settings
    parser.add_argument('--output_dir',
                        default='output/Pretrain/proclean_0324')
    parser.add_argument('--load_dir',
                        default='output/Pretrain/proclean_0324')
    parser.add_argument('--captioned_seq_sav_dir',
                        default='/datasets/uniref50_2018/proclean_0324')
    parser.add_argument('--captioned_ind_sav_dir',
                        default='caption_list/proclean_0324')
    parser.add_argument('--checkpoint',
                        default=None)

    ## system settings
    parser.add_argument('--use_wandb', default=False, type=bool)
    parser.add_argument('--wandb_project_name', default='AnnoDPO', type=str,
                        help='wandb project name')  # AnnoDPO
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_id', default=[0, 1], type=list)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    
    # Calculate effective batch size
    parser.add_argument('--effective_batch_size', default=4096, type=int,
                        help='Effective batch size across all GPUs')
    
    # Learning rate parameters
    parser.add_argument('--warmup_epochs', default=3, type=int,
                        help='Number of epochs for warmup')
    parser.add_argument('--warmup_lr', default=5e-7, type=float,
                        help='Initial learning rate for warmup')
    parser.add_argument('--init_lr', default=5e-5, type=float,
                        help='Peak learning rate after warmup')
    parser.add_argument('--min_lr', default=5e-7, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--lr_decay_rate', default=0.95, type=float,
                        help='Learning rate decay rate')
    
    parser.add_argument('--target_batch_size', default=4096, type=int,
                    help='Target global batch size after gradient accumulation')
    
    # PPO parameters
    parser.add_argument('--actor_lr', default=1e-5, type=float,
                        help='Learning rate for actor model')
    parser.add_argument('--critic_lr', default=1e-5, type=float,
                        help='Learning rate for critic model')
    parser.add_argument('--ppo_epochs', default=5, type=int,
                        help='Number of epochs to train PPO in each iteration')
    parser.add_argument('--reward_epoch', default=60, type=int,
                        help='Number of epochs to train reward model')
    parser.add_argument('--clip_ratio', default=0.2, type=float,
                        help='PPO clip ratio')
    parser.add_argument('--target_kl', default=0.01, type=float,
                        help='Target KL divergence')
    parser.add_argument('--value_clip_ratio', default=0.2, type=float,
                        help='Value function clip ratio')
    parser.add_argument('--max_epochs', default=20, type=int,
                        help='Maximum number of training epochs')
    parser.add_argument('--dpo_total_epochs', default=20, type=int,
                        help='Total number of training epochs')
    parser.add_argument('--dpo_epochs', default=0, type=int,
                        help='Current epoch to train DPO')
    # RLHF parameters
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--min_buffer_size', type=int, default=1000)
    parser.add_argument('--updates_per_step', type=int, default=8)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--soft_tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--max_kl', type=float, default=0.02)
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--priority_alpha', type=float, default=0.6,
                    help='Importance of priority (0 = uniform sampling, 1 = only sample high priority)')
    parser.add_argument('--priority_beta', type=float, default=0.4,
                    help='Starting value of IS weight (0 = no IS weight, 1 = fully compensate)')
    parser.add_argument('--beta_increment', type=float, default=0.001,
                    help='beta growth rate')
    
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.captioned_seq_sav_dir).mkdir(parents=True, exist_ok=True)
    Path(args.captioned_ind_sav_dir).mkdir(parents=True, exist_ok=True)
    main(args)
