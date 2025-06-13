import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import torch
import gc
from tqdm import tqdm
import torch.distributed as dist

def eval_go_classification(model, dataloader, args, epoch=0):
    # Use global variable to track eval count
    if not hasattr(eval_go_classification, 'eval_count'):
        eval_go_classification.eval_count = 0
    eval_go_classification.eval_count += 1
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Read GO classification files
    go_bp_df = pd.read_csv('go_biological_process.csv')
    go_cc_df = pd.read_csv('go_cellular_component.csv')
    go_mf_df = pd.read_csv('go_molecular_function.csv')
    
    # Extract GO indices
    bp_indices = set(go_bp_df['index'].values)
    cc_indices = set(go_cc_df['index'].values)
    mf_indices = set(go_mf_df['index'].values)
    
    # Initialize metrics dictionary
    metrics = {
        'TEST_total_batches': 0,
    }
    
    # Add evaluation metrics for three GO categories
    for category in ['BP', 'CC', 'MF']:
        metrics[f'TEST_anno_{category}_accuracy'] = 0.0
        metrics[f'TEST_anno_{category}_precision'] = 0.0
        metrics[f'TEST_anno_{category}_recall'] = 0.0
        metrics[f'TEST_anno_{category}_f1'] = 0.0
        metrics[f'TEST_anno_{category}_auroc'] = 0.0
        metrics[f'TEST_anno_{category}_fmax'] = 0.0
        metrics[f'TEST_anno_{category}_aupr'] = 0.0
        metrics[f'TEST_anno_{category}_count'] = 0

    model.eval()
    
    # Initialize category data collectors
    all_preds = {cat: [] for cat in ['BP', 'CC', 'MF']}
    all_true = {cat: [] for cat in ['BP', 'CC', 'MF']}
    seq_counts = {cat: 0 for cat in ['BP', 'CC', 'MF']}
    
    # Add tqdm progress bar, only display on main process
    dataloader_wrapper = tqdm(
        dataloader,
        desc=f'Evaluation #{eval_go_classification.eval_count}',
        dynamic_ncols=True,
        disable=args.rank != 0  # Disable progress bar on non-main processes
    )
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader_wrapper):
            input_seq, input_anno = batch
            input_seq = input_seq.contiguous().to(args.gpu).squeeze(1)
            input_anno = input_anno.contiguous().float().to(args.gpu)
            
            # Use autocast to reduce memory usage
            with torch.cuda.amp.autocast():
                outputs = model(input_seq, input_anno, "test", epoch)
            
            # Classify GO annotations
            anno_indices = torch.nonzero(input_anno, as_tuple=True)[1].cpu().numpy()  # Get GO indices for each batch
            
            # Determine which GO categories sequences belong to
            for b in range(input_anno.size(0)):
                # Get annotations for current sequence
                seq_anno = input_anno[b].cpu().numpy()
                seq_anno_indices = np.where(seq_anno == 1)[0]
                
                # Determine which categories it belongs to
                bp_present = any(idx in bp_indices for idx in seq_anno_indices)
                cc_present = any(idx in cc_indices for idx in seq_anno_indices)
                mf_present = any(idx in mf_indices for idx in seq_anno_indices)
                
                # Update category counts
                if bp_present: seq_counts['BP'] += 1
                if cc_present: seq_counts['CC'] += 1
                if mf_present: seq_counts['MF'] += 1
                
                # If there are predictions, collect category-specific predictions and ground truth
                if 'anno_logits' in outputs:
                    # Apply sigmoid to convert logits to probabilities
                    pred = torch.sigmoid(outputs['anno_logits'][b]).cpu().numpy()
                    true = seq_anno
                    
                    if bp_present:
                        # Keep only BP category predictions and ground truth
                        bp_mask = np.array([idx in bp_indices for idx in range(len(true))])
                        all_preds['BP'].append(pred[bp_mask])
                        all_true['BP'].append(true[bp_mask])
                    if cc_present:
                        # Keep only CC category predictions and ground truth
                        cc_mask = np.array([idx in cc_indices for idx in range(len(true))])
                        all_preds['CC'].append(pred[cc_mask])
                        all_true['CC'].append(true[cc_mask])
                    if mf_present:
                        # Keep only MF category predictions and ground truth
                        mf_mask = np.array([idx in mf_indices for idx in range(len(true))])
                        all_preds['MF'].append(pred[mf_mask])
                        all_true['MF'].append(true[mf_mask])
            
            metrics['TEST_total_batches'] += 1
    
    # Calculate evaluation metrics for each category
    for category in ['BP', 'CC', 'MF']:
        if all_preds[category] and all_true[category]:
            # Convert to numpy arrays
            cat_preds = np.vstack(all_preds[category])
            cat_true = np.vstack(all_true[category])
            
            # Calculate evaluation metrics for each category
            # Use threshold 0.5 to calculate basic metrics
            preds_binary = (cat_preds > 0.5).astype(int)
            metrics[f'TEST_anno_{category}_precision'] = precision_score(cat_true, preds_binary, average='micro', zero_division=0)
            metrics[f'TEST_anno_{category}_recall'] = recall_score(cat_true, preds_binary, average='micro', zero_division=0)
            metrics[f'TEST_anno_{category}_f1'] = f1_score(cat_true, preds_binary, average='micro', zero_division=0)
            
            # Calculate AUROC
            try:
                # Handle case with only one class
                metrics[f'TEST_anno_{category}_auroc'] = roc_auc_score(cat_true, cat_preds, average='micro')
            except ValueError:
                metrics[f'TEST_anno_{category}_auroc'] = 0.0
            
            # Calculate F1-max
            precisions, recalls, thresholds = precision_recall_curve(cat_true.flatten(), cat_preds.flatten())
            f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
            metrics[f'TEST_anno_{category}_fmax'] = np.max(f1_scores)

            # Calculate AUPR
            metrics[f'TEST_anno_{category}_aupr'] = auc(recalls, precisions)

    # Synchronize sequence counts
    for cat in ['BP', 'CC', 'MF']:
        count_tensor = torch.tensor(seq_counts[cat], device=args.gpu)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        seq_counts[cat] = count_tensor.item()

    # Synchronize metrics across all processes
    world_size = dist.get_world_size()
    for key in metrics:
        tensor = torch.tensor(metrics[key], device=args.gpu)
        
        if key == 'TEST_total_batches':
            # Sum batch counts
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif key.startswith('TEST_anno_'):
            if key.endswith('_count'):
                # Sum counts
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            else:
                # For evaluation metrics, sum first then average
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor = tensor / world_size
        else:
            # Default sum for other metrics
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        metrics[key] = tensor.item()

    # Clean up after evaluation
    torch.cuda.empty_cache()
    gc.collect()

    # Print GO category statistics
    if args.rank == 0:
        print("\n" + "="*50)
        print("GO Category Statistics:")
        total_sequences = metrics['TEST_total_batches'] * dataloader.batch_size  # Estimate total number of sequences
        print(f"Total sequences: {total_sequences}")
        for cat in ['BP', 'CC', 'MF']:
            percentage = (seq_counts[cat] / total_sequences) * 100  # Corrected percentage calculation
            print(f"{cat} category sequences: {seq_counts[cat]} ({percentage:.2f}%)")
        print("="*50)
        
        print("\nGO Category Evaluation Metrics:")
        for cat in ['BP', 'CC', 'MF']:
            print(f"\n{cat} Category Performance:")
            print(f"Precision: {metrics[f'TEST_anno_{cat}_precision']:.4f}")
            print(f"Recall: {metrics[f'TEST_anno_{cat}_recall']:.4f}")
            print(f"F1 Score: {metrics[f'TEST_anno_{cat}_f1']:.4f}")
            print(f"AUROC: {metrics[f'TEST_anno_{cat}_auroc']:.4f}")
            print(f"F1-max: {metrics[f'TEST_anno_{cat}_fmax']:.4f}")
            print(f"AUPR: {metrics[f'TEST_anno_{cat}_aupr']:.4f}")
        print("="*50 + "\n")
    
    return metrics
