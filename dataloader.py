import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import esm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import os
import sys
import torch.distributed as dist

class Dataset_finetune_untokenized(data.Dataset):
    def __init__(self, model, num_annotation=7533, max_len=512):
        super(Dataset_finetune_untokenized, self).__init__()
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.model = model
        self.max_len = max_len
        self.n_annotation = num_annotation
        self.sw_path = f"/datasets/swissprot/swissprot_train_untokenized_new.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_train_count = f[self.seqs].shape[0]
        print("sw_train_count", self.sw_train_count)

    def process_seq(self, seq):
        seq_len = len(seq)
        if seq_len < self.max_len:
            # If sequence length is less than max_len, pad with '-'
            padded_seq = seq + '-' * (self.max_len - seq_len)
            return padded_seq
        elif seq_len > self.max_len:
            # If sequence length is greater than max_len, randomly select a segment of length max_len
            start_idx = np.random.randint(0, seq_len - self.max_len + 1)
            return seq[start_idx:start_idx + self.max_len]
        else:
            return seq

    def __len__(self):
        return self.sw_train_count

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.sw_path, 'r') as sw_datasets:
            seq = sw_datasets[self.seqs][idx].decode("utf-8")
            kws_indices = sw_datasets[self.kws][idx]

        # Process sequence
        seq = self.process_seq(seq)

        if np.any(kws_indices):
            kws[kws_indices] = True

        return seq, torch.tensor(kws, dtype=torch.float)

    def collate_fn(self, batch):
        seqs, kws = zip(*batch)
        tokenized_seqs = self.model._tokenize(list(seqs))
        kws_tensor = torch.stack(kws)
        
        if tokenized_seqs.is_cuda:
            tokenized_seqs = tokenized_seqs.cpu()
            
        return tokenized_seqs, kws_tensor


class Dataset_swiss_random_test_untokenized(data.Dataset):
    def __init__(self, model, num_annotation=7533, max_len=512):
        super(Dataset_swiss_random_test_untokenized, self).__init__()
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.model = model
        self.max_len = max_len
        self.n_annotation = num_annotation
        self.sw_path = f"/datasets/swissprot/swissprot_test_untokenized_new.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_test_count = f[self.seqs].shape[0]
        print("sw_test_count", self.sw_test_count)

    def process_seq(self, seq):
        seq_len = len(seq)
        if seq_len < self.max_len:
            # If sequence length is less than max_len, pad with '-'
            padded_seq = seq + '-' * (self.max_len - seq_len)
            return padded_seq
        elif seq_len > self.max_len:
            # If sequence length is greater than max_len, randomly select a segment of length max_len
            start_idx = np.random.randint(0, seq_len - self.max_len + 1)
            return seq[start_idx:start_idx + self.max_len]
        else:
            return seq

    def __len__(self):
        return self.sw_test_count

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.sw_path, 'r') as sw_datasets:
            seq = sw_datasets[self.seqs][idx].decode("utf-8")
            kws_indices = sw_datasets[self.kws][idx]

        # Process sequence
        seq = self.process_seq(seq)

        if np.any(kws_indices):
            kws[kws_indices] = True

        return seq, torch.tensor(kws, dtype=torch.float)

    def collate_fn(self, batch):
        seqs, kws = zip(*batch)
        tokenized_seqs = self.model._tokenize(list(seqs))
        kws_tensor = torch.stack(kws)
        
        if tokenized_seqs.is_cuda:
            tokenized_seqs = tokenized_seqs.cpu()
            
        return tokenized_seqs, kws_tensor


class Dataset_uniref50_caption_new(data.Dataset):
    def __init__(self, model=None, num_annotation=7533, max_len=512, epoch=0, captioned_seq_sav_dir=None):
        super(Dataset_uniref50_caption_new, self).__init__()
        # Basic attribute settings
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.inds = 'inds'
        self.prot_ids = 'prot_ids'
        self.model = model
        self.max_len = max_len
        self.n_annotation = num_annotation
        
        # Select data path based on epoch
        if epoch == 0:
            self.data_path = '/datasets/uniref50_2018/uniref50_2018_new_split_sample_untokenized.hdf5'
        else:
            self.data_path = captioned_seq_sav_dir
            
        # Get dataset size
        try:
            with h5py.File(self.data_path, 'r') as f:
                self.data_count = f[self.seqs].shape[0]
            print(f"Dataset size: {self.data_count}")
        except Exception as e:
            print(f"Error opening dataset file: {e}")
            raise

        # Save original model's tokenize method and data type
        if model is not None:
            self.tokenize_fn = model._tokenize
        else:
            self.tokenize_fn = None

    def process_seq(self, seq):
        seq_len = len(seq)
        if seq_len < self.max_len:
            # If sequence length is less than max_len, pad with '-'
            padded_seq = seq + '-' * (self.max_len - seq_len)
            return padded_seq
        elif seq_len > self.max_len:
            # If sequence length is greater than max_len, randomly select a segment
            start_idx = np.random.randint(0, seq_len - self.max_len + 1)
            return seq[start_idx:start_idx + self.max_len]
        else:
            return seq

    def __len__(self):
        return self.data_count

    def __getitem__(self, idx):
        try:
            # Initialize annotation vector
            kws = np.zeros((self.n_annotation), dtype=np.float32)  # Use float32
            
            # Read data
            with h5py.File(self.data_path, 'r') as datasets:
                seq = datasets[self.seqs][idx].decode("utf-8")
                kws_indices = datasets[self.kws][idx]
                ind = datasets[self.inds][idx]
                prot_id = datasets[self.prot_ids][idx]
            
            # Process annotation
            if np.any(kws_indices):
                kws[kws_indices] = 1.0  # Use 1.0 instead of True
            
            # Process sequence
            seq = self.process_seq(seq)
            
            # Return all required information, ensure data type consistency
            return (
                torch.tensor(kws, dtype=torch.float32),  # Use float32
                torch.tensor(len(seq), dtype=torch.int64),  # Use int64
                torch.tensor(ind, dtype=torch.int64),
                torch.tensor(idx, dtype=torch.int64),
                prot_id,
                seq
            )
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            raise

    def collate_fn(self, batch):
        # Unpack batch data
        kws, seq_lens, inds, idxs, prot_ids, seqs = zip(*batch)
        
        # If tokenize_fn exists, use it to process sequence, and ensure it is long integer
        if self.tokenize_fn is not None:
            tokenized_seqs = self.tokenize_fn(list(seqs))
            if tokenized_seqs.is_cuda:
                tokenized_seqs = tokenized_seqs.cpu()
            # Ensure tokenized_seqs is long integer
            tokenized_seqs = tokenized_seqs.long()
        else:
            tokenized_seqs = seqs
            
        # Stack tensor data, ensure data type consistency
        kws_tensor = torch.stack(kws).to(dtype=torch.float32)
        seq_lens_tensor = torch.stack(seq_lens)
        inds_tensor = torch.stack(inds)
        idxs_tensor = torch.stack(idxs)
        
        return (
            kws_tensor,
            seq_lens_tensor,
            inds_tensor, 
            idxs_tensor,
            prot_ids,
            tokenized_seqs,  # Now ensure it is long integer
            seqs
        )

class Dataset_swissprot(data.Dataset):
    def __init__(self, num_annotation=7533, max_len=512, mode="train", model=None):
        super(Dataset_swissprot, self).__init__()
        self.max_len = max_len
        self.n_annotation = num_annotation
        self.model = model
        if mode == "train":
            self.sw_path = "/datasets/swissprot_previous_versions/2020_01/proteins_train.h5"
        elif mode == "test":
            self.sw_path = "/datasets/swissprot_previous_versions/2020_01/proteins_test.h5"
        elif mode == "caption_rlhf":
            self.sw_path = "/datasets/swissprot_previous_versions/2020_01/new_proteins_2024_11_compare_to_2020_01.h5"
        elif mode.startswith("eval_freq_") or mode.startswith("zeroshot_freq_"):
            self.sw_path = f"/datasets/swissprot_previous_versions/2020_01/frequency_datasets/{mode.split('_')[2]}_freq_proteins.h5"

        # Count total number of proteins in the HDF5 file
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_count = len(f.keys())  # Count number of groups
        print(f"{mode} dataset protein count: {self.sw_count}")
        
        # Store all group names for later access
        with h5py.File(self.sw_path, 'r') as f:
            self.group_names = list(f.keys())

    def process_seq(self, seq):
        seq_len = len(seq)
        if seq_len < self.max_len:
            padded_seq = seq + '-' * (self.max_len - seq_len)
            return padded_seq
        elif seq_len > self.max_len:
            start_idx = np.random.randint(0, seq_len - self.max_len + 1)
            return seq[start_idx:start_idx + self.max_len]
        else:
            return seq

    def __len__(self):
        return self.sw_count

    def __getitem__(self, idx):
        with h5py.File(self.sw_path, 'r') as f:
            group_name = self.group_names[idx]
            group = f[group_name]
            
            # Get sequence from SQ dataset
            seq = group['SQ'][()].decode('utf-8')
            
            # Get GO annotations and convert to tensor
            go_binary = group['GO_binary'][()]
            go_binary_tensor = torch.tensor(go_binary, dtype=torch.float)
            
            # Print debug information
            # print("go_binary_tensor shape", go_binary_tensor.shape)
            # print("indices of 1 in tensor", torch.where(go_binary_tensor == 1)[0])
            
            # Process sequence if needed
            seq = self.process_seq(seq)
            
            return seq, go_binary_tensor

    def collate_fn(self, batch):
        seqs, kws = zip(*batch)
        tokenized_seqs = self.model._tokenize(list(seqs))
        kws_tensor = torch.stack(kws)
        
        if tokenized_seqs.is_cuda:
            tokenized_seqs = tokenized_seqs.cpu()
            
        return tokenized_seqs, kws_tensor

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank,
                                                  shuffle=shuffles)
    return sampler

def worker_init_fn(worker_id):
    """Initialize data loader worker"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Set device for worker
        device_id = worker_info.id % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        
        # Set random seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

def create_loader(dataset, sampler, batch_size, num_workers, is_training, collate_fn=None, worker_init_fn=worker_init_fn, generator=None, prefetch_factor=3, pin_memory=True, persistent_workers=True):
    """
    Optimized data loader creation function
    
    Args:
        dataset: dataset
        sampler: sampler
        batch_size: batch size
        num_workers: number of workers (will be automatically optimized)
        is_training: whether in training mode
        collate_fn: data collation function
        worker_init_fn: worker initialization function
        generator: random number generator
    """
    try:

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else None,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            sampler=sampler,
            shuffle=(sampler is None) and is_training,
            drop_last=is_training,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=generator,
            multiprocessing_context='spawn'
        )
        
        return loader
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        raise