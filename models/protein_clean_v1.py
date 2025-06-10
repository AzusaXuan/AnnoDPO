import sys
from models.blip_protein import create_proteinBERT
import torch
from torch import nn
import torch.nn.functional as F
import utils
from torchvision.ops.focal_loss import sigmoid_focal_loss as focal_loss
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall, BinaryPrecision, AUROC
import numpy as np

class CrossModalPredictor(nn.Module):
    def __init__(self, seq_dim, n_annotation, dropout=0.1, num_residual_blocks=2):
        super().__init__()
        
        # 1. Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(seq_dim, seq_dim),  # Merge three views of features
            nn.LayerNorm(seq_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Residual blocks
        # self.residual_blocks = nn.ModuleList([
        #     ResidualBlock(
        #         seq_dim=seq_dim,
        #         dropout=dropout
        #     ) for _ in range(num_residual_blocks)
        # ])
        
        # 3. Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(seq_dim, n_annotation)
        )
        
        # 4. Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, seq_feats, anno_embeds=None):
        # 1. Feature concatenation
        combined_features = torch.cat([
            seq_feats['cls'],
            # seq_feats['mean'],
            # seq_feats['max']
        ], dim=-1)
        
        # 2. Feature fusion
        features = self.feature_fusion(combined_features)
        
        # 3. Residual block processing
        # for block in self.residual_blocks:
        #     features = block(features)
        
        # 4. Prediction
        logits = self.prediction_head(features)
        
        return logits

class ResidualBlock(nn.Module):
    def __init__(self, seq_dim, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(seq_dim, seq_dim),
            nn.LayerNorm(seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim, seq_dim)
        )
        self.norm = nn.LayerNorm(seq_dim)
    
    def forward(self, x):
        residual = x
        x = self.mlp(x)
        x = x + residual
        x = self.norm(x)
        return x

def compute_diversity_loss(features):
    """Compute feature diversity loss"""
    # Compute similarity matrix between features
    sim_matrix = torch.mm(features, features.t())
    # Remove diagonal (self-similarity)
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    sim_matrix = sim_matrix * (1 - mask)
    # Compute feature diversity loss
    return sim_matrix.pow(2).mean()

def compute_orthogonality_loss(features):
    """Compute feature orthogonality loss"""
    # Compute orthogonality between features
    gram_matrix = torch.mm(features.t(), features)
    identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
    return F.mse_loss(gram_matrix, identity)

class Protein_Clean_Seq_Anno_esmc(nn.Module):
    def __init__(self, seq_model=None, args=None):
        super().__init__()
        # Save args as class attribute
        self.args = args
        
        # Set default data type to float32
        torch.set_default_dtype(torch.float32)
        
        self.seq_encoder = seq_model
        args.dim = self.seq_encoder.embed.embedding_dim
        self.use_itc = args.use_itc
        self.use_nitc = args.use_nitc
        self.use_alm = args.use_alm
        self.use_wandb = args.use_wandb
        self.seq_bert_dim = args.dim # 960
        self.anno_bert_dim = args.dim_global # 512
        self.n_annotation = args.num_annotation
        
        self.anno_encoder = create_proteinBERT("anno_bert_itc", args)
        self.anno_proj = nn.Linear(self.anno_bert_dim, self.seq_bert_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        self.itm_head = nn.Linear(self.anno_bert_dim, 2)
        
        self.to_annotation_logits = nn.Linear(args.dim_global, self.n_annotation)

        # clip
        self.labels = None
        self.noisy_labels = None
        self.last_local_batch_size = None
        self.n_noise = 5  # Number of noise versions generated for each sample
        self.expanded_noisy_labels = None
        self.noise_weight = 0.01
        self.mask_rate = 1.0
        # adaptive loss
        if args.actual_epoch > 0 and args.mode == "train":
            self.use_adaptive_loss = True
        else:
            self.use_adaptive_loss = False

        # Save original train method
        self.original_train = self.train
        
        # Override train method
        self.train = self.new_train

        # Enhanced sequence feature extractor
        self.seq_proj = nn.Sequential(
            # First layer: Expand feature dimension
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim * 2),
            nn.LayerNorm(self.seq_bert_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # Second layer: Feature compression
            nn.Linear(self.seq_bert_dim * 2, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            # Final projection
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim)
        )

        # Enhanced annotation feature extractor
        self.anno_proj = nn.Sequential(
            # First layer: Expand feature dimension
            nn.Linear(self.anno_bert_dim, self.seq_bert_dim * 2),
            nn.LayerNorm(self.seq_bert_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # Second layer: Feature compression
            nn.Linear(self.seq_bert_dim * 2, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            # Final projection
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim)
        )
        
        # Feature alignment
        self.alignment_layer = nn.Sequential(
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim)
        )

        # Instantiate new predictor
        self.anno_predictor = CrossModalPredictor(
            seq_dim=self.seq_bert_dim,      # ESM dimension 960
            n_annotation=self.n_annotation,
            dropout=args.predictor_dropout,
            num_residual_blocks=args.num_residual_blocks
        )
        
        # Initialize evaluation metrics
        self.metrics = {
            'accuracy': BinaryAccuracy(threshold=0.5),
            'precision': BinaryPrecision(threshold=0.5),
            'recall': BinaryRecall(threshold=0.5),
            'auroc': AUROC(task="binary"),
            'f1': BinaryF1Score(threshold=0.5)
        }

    def new_train(self, mode=True):
        """
        Override train method, control seq_encoder training status
        """
        # Call original train method
        self.original_train(mode)
        
        # Check if args exists
        if not hasattr(self, 'args'):
            print("Warning: args not found, using default behavior")
            return self
        
        # Freeze seq_encoder only when actual_epoch==0
        if self.args.actual_epoch == 0:
            # Force seq_encoder to remain in eval mode
            self.seq_encoder.eval()
            # Ensure gradients are not computed
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
        else:
            # Allow training seq_encoder in other epochs
            if mode:  # If training mode
                self.seq_encoder.train()
                for param in self.seq_encoder.parameters():
                    param.requires_grad = True
        return self

    def compute_intra_sac_loss(self, seq_feat, combined_anno_feat):
        # 1. Compute similarity
        logits = torch.einsum('bd,nbd->bn', seq_feat, combined_anno_feat)  # [bs, 1+n_noise]
        logits = logits / self.temp
        
        # 2. Compute probability
        probs = torch.sigmoid(logits)  # [bs, 1+n_noise]
        
        # 3. Compute pt (probability for target class)
        pt = torch.where(self.noisy_labels == 1, probs, 1 - probs)
        
        # 4. Set focal loss parameters
        alpha = 0.25  # Positive sample weight
        gamma = 2.0   # Modulation factor
        
        # 5. Compute class weights
        alpha_weight = torch.where(
            self.noisy_labels == 1,
            torch.ones_like(logits) * alpha,
            torch.ones_like(logits) * (1 - alpha)
        )
        
        # 6. Compute focal weight
        focal_weight = (1 - pt).pow(gamma)
        
        # 7. Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            self.noisy_labels,
            reduction='none'
        )
        
        # 8. Combine focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()

    def forward(self, protein_seq, anno, mode="train", epoch=0, mask=None):
        # Add input check
        if protein_seq is None:
            raise ValueError("protein_seq cannot be None")
        
        bs = protein_seq.size(0)
        device = protein_seq.device

        # Save original anno for evaluation and training
        original_anno = anno.to(device, dtype=torch.float32, non_blocking=True)

        # Initialize return value dictionary
        loss_dict = {
            'itc_loss': torch.tensor(0.0, device=device),
            'intra_sac_loss': torch.tensor(0.0, device=device),
            'pred_loss': torch.tensor(0.0, device=device),
            'total_loss': torch.tensor(0.0, device=device),
            'diversity_loss': torch.tensor(0.0, device=device),
            'seq_diversity_loss': torch.tensor(0.0, device=device),
            'anno_diversity_loss': torch.tensor(0.0, device=device),
            'orthogonality_loss': torch.tensor(0.0, device=device),
        }
        metrics_dict = {}
        
        # Ensure input is on correct device
        encoder_input_seq = protein_seq.to(device, dtype=torch.long, non_blocking=True)
        encoder_input_anno = original_anno  # Use original anno
        
        # Use ESM model to process sequence
        with torch.cuda.amp.autocast(enabled=True):
            try:
                # Determine whether to compute gradients based on training stage
                if mode == "train":
                    # Training mode and not frozen stage, allow gradient computation
                    output = self.seq_encoder(encoder_input_seq)
                else:
                    # Evaluation mode or frozen stage, do not compute gradients
                    with torch.no_grad():
                        output = self.seq_encoder(encoder_input_seq)
                        
                seq_logits, seq_hidden_embeds, hiddens = (
                    output.sequence_logits,
                    output.embeddings,
                    output.hidden_states,
                )
            except RuntimeError as e:
                print(f"Error in seq_encoder: {e}")
                raise e

        # Ensure all tensors are on the same device and have correct type
        seq_hidden_embeds = seq_hidden_embeds.to(device, dtype=torch.float32)
        
        # Extract sequence features
        seq_views = {
            'cls': seq_hidden_embeds[:, 0],
        }
        
        # Project and align each view
        seq_feats = {}
        for k, v in seq_views.items():
            # 1. Basic projection
            proj_feat = self.seq_proj(v)
            # 2. Feature alignment
            aligned_feat = self.alignment_layer(proj_feat)
            # 3. Normalization
            seq_feats[k] = F.normalize(aligned_feat, dim=-1) # [bs, seq_bert_dim]
        
        # Get annotation features and align
        anno_embeds = self.anno_encoder(encoder_input_anno, mask)
        anno_feat = self.anno_proj(anno_embeds) # [bs, seq_bert_dim]
        aligned_anno_feat = self.alignment_layer(anno_feat)
        anno_feat = F.normalize(aligned_anno_feat, dim=-1)
        # Collect features on all GPUs
        anno_feat_all, seq_feats_all = {}, {}
        for view_name, seq_feat in seq_feats.items():
            anno_feat_all[view_name], seq_feats_all[view_name] = utils.all_gather_batch(
                [anno_feat, seq_feat]
            )
        similarity_all = {}
        for view_name in seq_feats:
            # Compute similarity matrix
            similarity = seq_feats_all[view_name] @ anno_feat_all[view_name].t()
            similarity_all[view_name] = similarity # [bs*world_size, bs*world_size]
        ###============== Sequence-annotation contrastive ===================###
        if self.use_itc or self.use_nitc:
            try:
                with torch.no_grad():
                    self.temp.clamp_(0.07, 0.5)
                if bs != self.last_local_batch_size:
                        self.labels = bs * utils.get_rank() + torch.arange(
                            bs, device=device
                        )
                        self.last_local_batch_size = bs
                if mode == "test":
                        
                    correct_matches = 0
                    total_matches = 0
                    
                    # Add top-k accuracy statistics
                    k_values = [1, 3, 5, 10]  # Define k values to compute
                    topk_correct = {k: 0 for k in k_values}
                    
                    # # Collect features on all GPUs
                    # anno_feat_all, seq_feats_all = {}, {}
                    # for view_name, seq_feat in seq_feats.items():
                    #     anno_feat_all[view_name], seq_feats_all[view_name] = utils.all_gather_batch(
                    #         [anno_feat, seq_feat]
                    #     )
                    
                    # Compute similarity and matching accuracy for each view
                    for view_name in seq_feats:
                        # Compute similarity matrix
                        # similarity = seq_feats_all[view_name] @ anno_feat_all[view_name].t()
                        similarity = similarity_all[view_name]
                        # Compute feature alignment metric
                        alignment_score = F.cosine_similarity(
                            seq_feats_all[view_name],
                            anno_feat_all[view_name],
                            dim=-1
                        ).mean()
                        
                        metrics_dict[f'alignment_score_{view_name}'] = alignment_score
                        
                        # Compute top-k accuracy
                        for k in k_values:
                            # Get top-k annotation indices
                            _, topk_indices = similarity.topk(k=k, dim=1)
                            # Check if correct label is in top-k
                            target_indices = torch.arange(len(topk_indices), device=device).unsqueeze(1)
                            correct_topk = (topk_indices == target_indices).any(dim=1).sum().item()
                            topk_correct[k] += correct_topk
                            # Record top-k accuracy for each view
                            metrics_dict[f'itc_accuracy_{view_name}_top{k}'] = torch.tensor(
                                correct_topk / len(topk_indices),
                                device=device
                            )
                        
                        # Get top-1 annotation index for each sequence
                        pred_indices = similarity.argmax(dim=1)
                        
                        # Compute number of correct matches
                        correct = (pred_indices == torch.arange(len(pred_indices), device=device)).sum()
                        correct_matches += correct.item()
                        total_matches += len(pred_indices)
                        
                        metrics_dict[f'itc_accuracy_{view_name}'] = torch.tensor(
                            correct.item() / len(pred_indices),
                            device=device
                        )
                    
                    # Record overall matching accuracy
                    metrics_dict['itc_accuracy_total'] = torch.tensor(
                        correct_matches / total_matches,
                        device=device
                    )
                    
                    # Record overall top-k accuracy
                    for k in k_values:
                        metrics_dict[f'itc_accuracy_total_top{k}'] = torch.tensor(
                            topk_correct[k] / total_matches,
                            device=device
                        )
                    
                    # Compute ITC loss using predicted anno
                    with torch.no_grad():
                        anno_logits = self.anno_predictor(
                            seq_feats=seq_feats,
                            anno_embeds=None
                        )
                        pred_anno = torch.sigmoid(anno_logits)
                        pred_anno = (pred_anno > 0.5).float()
                        
                        # Get features of predicted anno
                        pred_anno_embeds = self.anno_encoder(pred_anno, mask)
                        pred_anno_feat = self.anno_proj(pred_anno_embeds)
                        pred_anno_feat = self.alignment_layer(pred_anno_feat)
                        pred_anno_feat = F.normalize(pred_anno_feat, dim=-1)
                        
                        # Compute ITC loss using predicted anno
                        test_loss_ita = 0
                        for view_name, seq_feat in seq_feats.items():
                            pred_anno_feat_all, seq_feat_all = utils.all_gather_batch([pred_anno_feat, seq_feat])
                            test_loss_ita += (F.cross_entropy(seq_feat @ pred_anno_feat_all.t() / self.temp, self.labels) + 
                                            F.cross_entropy(pred_anno_feat @ seq_feat_all.t() / self.temp, self.labels)) / 2
                        
                        loss_dict['itc_loss'] = test_loss_ita / len(seq_feats)
                        loss_dict['total_loss'] += loss_dict['itc_loss']
                
                elif mode == "train" or mode == "rlhf":
                    if self.use_nitc:
                        # 1. Create noisy anno
                        masks = (torch.rand(self.n_noise, *anno.shape, device=device) > self.mask_rate).float()
                        non_zero_mask = (anno != 0).float()
                        non_zero_mask = non_zero_mask.unsqueeze(0).expand(self.n_noise, -1, -1)
                        noisy_annos = anno.unsqueeze(0) * masks * non_zero_mask  # [n_noise, bs, n_anno]
                        # noisy anno features
                        noisy_annos_flat = noisy_annos.view(-1, anno.size(-1)) # [n_noise*bs, n_anno]
                        noisy_anno_embeds = self.anno_encoder(noisy_annos_flat, mask)
                        noisy_anno_feat = self.anno_proj(noisy_anno_embeds)
                        noisy_anno_feat = self.alignment_layer(noisy_anno_feat)
                        noisy_anno_feat = F.normalize(noisy_anno_feat, dim=-1).view(self.n_noise, bs, -1) # [n_noise, bs, seq_bert_dim]
                        # Create labels
                        # Compare original anno and noisy_annos
                        anno_expanded = anno.unsqueeze(0)  # [1, bs, n_classes]
                        # Compute similarity between each noisy anno and original anno
                        matches = torch.all(anno_expanded == noisy_annos, dim=-1)  # [n_noise, bs]
                        matches = matches.transpose(0, 1)  # [bs, n_noise]
                        
                        # Create label matrix
                        self.noisy_labels = torch.zeros(bs, 1+self.n_noise, device=device)
                        self.noisy_labels[:, 0] = 1  # Original anno position is positive sample
                        self.noisy_labels[:, 1:] = matches  # Noisy_annos with same original anno are also positive samples
                        
                    # 2. Compute ITC loss
                    loss_ita = 0
                    for view_name, seq_feat in seq_feats.items():
                        # Collect all features
                        anno_feat_all, seq_feat_all = utils.all_gather_batch([anno_feat, seq_feat]) # [bs*world_size, seq_bert_dim], [bs*world_size, seq_bert_dim]
                        if self.use_nitc:
                            loss_intra_sac = 0

                            # 2. Merge original anno and noisy_anno
                            # Expand anno_feat to [bs, 1, dim] and transpose to [1, bs, dim]
                            expanded_anno_feat = anno_feat.unsqueeze(0)  # [1, bs, dim]
                            # Concatenate all anno features: [1+n_noise, bs, dim]
                            combined_anno_feat = torch.cat([expanded_anno_feat, noisy_anno_feat], dim=0)
                            
                            # 3. Compute focal loss
                            intra_sac_loss = self.compute_intra_sac_loss(seq_feat, combined_anno_feat)
                            loss_dict['intra_sac_loss'] = intra_sac_loss
                            loss_dict['total_loss'] += self.noise_weight * loss_dict['intra_sac_loss']

                    # Compute sequence-annotation contrastive loss
                    loss_s2a = F.cross_entropy(seq_feat @ anno_feat_all.t() / self.temp, self.labels.long())
                    
                    # Compute annotation-sequence contrastive loss
                    loss_a2s = F.cross_entropy(anno_feat @ seq_feat_all.t() / self.temp, self.labels.long())
                    

                    # Merge bidirectional loss
                    loss_ita += (loss_s2a + loss_a2s) / 2
                    
                    loss_dict['itc_loss'] = loss_ita / len(seq_feats)
                    loss_dict['total_loss'] += loss_dict['itc_loss']
                
                    # Compute feature regularization loss
                    seq_diversity_loss = 0
                    for view_name, seq_feat in seq_feats.items():
                        seq_diversity_loss += compute_diversity_loss(seq_feat)
                    seq_diversity_loss = seq_diversity_loss / len(seq_feats)

                    # Compute annotation feature diversity loss
                    anno_diversity_loss = compute_diversity_loss(anno_feat)

                    # Compute orthogonality loss
                    orthogonality_loss = 0
                    for view_name, seq_feat in seq_feats.items():
                        orthogonality_loss += compute_orthogonality_loss(seq_feat)
                    orthogonality_loss = orthogonality_loss / len(seq_feats)
                    
                    # Combine all regularization losses, using smaller weights
                    diversity_weight = 0.1
                    orthogonality_weight = 0.1
                    regularization_loss = (
                        diversity_weight * (seq_diversity_loss + anno_diversity_loss) +
                        orthogonality_weight * orthogonality_loss
                    )

                    # Update total loss
                    loss_dict['diversity_loss'] = regularization_loss
                    loss_dict['total_loss'] += regularization_loss

                    # Record individual loss components
                    metrics_dict.update({
                        'seq_diversity_loss': seq_diversity_loss,
                        'anno_diversity_loss': anno_diversity_loss,
                        'orthogonality_loss': orthogonality_loss
                    })
                    
            except Exception as e:
                print(f"Error in ITC processing: {e}")
                raise e

        ###============== Annotation Learning Module ===================###
        if self.use_alm:
            try:
                # Test mode has already computed anno_logits, training and validation mode need to compute
                if mode == "train":
                    anno_logits = self.anno_predictor(
                        seq_feats=seq_feats,
                        anno_embeds=None
                    )
                    pred_anno = torch.sigmoid(anno_logits)
                    pred_anno = (pred_anno > 0.5).float()
                    # Use binary cross entropy loss
                    loss_dict['pred_loss'] = F.binary_cross_entropy_with_logits(
                        anno_logits, 
                        original_anno
                    )
                    
                    # Update total_loss
                    loss_dict['total_loss'] += loss_dict['pred_loss']
                
                # Compute evaluation metrics and pred_loss in all modes
                elif mode == "test":
                    with torch.no_grad():
                        anno_logits = self.anno_predictor(
                            seq_feats=seq_feats,
                            anno_embeds=None
                        )
                        # Use previously computed anno_logits in test mode
                        anno_pred = torch.sigmoid(anno_logits)
                        pred_anno = (anno_pred > 0.5).float()
                        # Compute pred_loss
                        loss_dict['pred_loss'] = F.binary_cross_entropy_with_logits(
                            anno_logits,
                            original_anno
                        )
                        
                        # Update total_loss
                        loss_dict['total_loss'] += loss_dict['pred_loss']
                            
                        # Compute other metrics
                        for metric_name, metric in self.metrics.items():
                            metrics_dict[f'anno_{metric_name}'] = torch.tensor(
                                metric(anno_pred.cpu(), original_anno.cpu()),
                                device=device
                            )
                        
                        # Compute F-max
                        fmax = 0.0
                        for threshold in (t/20 for t in range(21)):
                            f1_score = BinaryF1Score(threshold=threshold)
                            fmax = max(fmax, f1_score(anno_pred.cpu(), original_anno.cpu()))
                        metrics_dict['anno_fmax'] = torch.tensor(fmax, device=device)
                        
                        # Compute AUPRC
                        try:
                            from sklearn.metrics import average_precision_score
                            auprc = average_precision_score(
                                original_anno.cpu().numpy(),
                                anno_pred.detach().cpu().numpy(),
                                average='micro'
                            )
                            metrics_dict['anno_auprc'] = torch.tensor(auprc, device=device)
                        except ImportError:
                            print("sklearn not found, skipping AUPRC calculation")
                
            except Exception as e:
                print(f"Error in ALM processing: {e}")
                raise e
        
        # If neither ITC nor ALM is used, total_loss is already 0
        # Merge all return values
        return {
            **loss_dict,
            **metrics_dict,
            'anno_logits': anno_logits,
            'seq_logits': seq_logits,
            'similarity': similarity_all['cls'],
            'pred_anno': pred_anno,
        }
    
    def get_anno_logits(self, protein_seq):
        """
        Get logits of annotation probabilities
        
        Args:
            protein_seq: Input annotation, shape [batch_size, n_seq]
        """
        # Add input check
        if protein_seq is None:
            raise ValueError("protein_seq cannot be None")
        
        bs = protein_seq.size(0)
        device = protein_seq.device

        # Ensure input is on correct device
        encoder_input_seq = protein_seq.to(device, dtype=torch.long, non_blocking=True)
        
        # Use ESM model to process sequence
        with torch.cuda.amp.autocast(enabled=True):
            try:
                with torch.no_grad():  # Ensure no gradient computation
                    output = self.seq_encoder(encoder_input_seq)
                    logits, seq_hidden_embeds, hiddens = (
                        output.sequence_logits,
                        output.embeddings,
                        output.hidden_states,
                    )
            except RuntimeError as e:
                print(f"Error in seq_encoder: {e}")
                raise e

        # Ensure all tensors are on the same device and have correct type
        seq_hidden_embeds = seq_hidden_embeds.to(device, dtype=torch.float32)
        
        # Extract sequence features
        seq_views = {
            'cls': seq_hidden_embeds[:, 0],
        }
        
        # Project and align each view
        seq_feats = {}
        for k, v in seq_views.items():
            # 1. Basic projection
            proj_feat = self.seq_proj(v)
            # 2. Feature alignment
            aligned_feat = self.alignment_layer(proj_feat)
            # 3. Normalization
            seq_feats[k] = F.normalize(aligned_feat, dim=-1)
        
        # Get logits
        anno_logits = self.anno_predictor(
                        seq_feats=seq_feats,
                        anno_embeds=None
                    )
        return anno_logits

def protein_clean_seq_anno_esmc(**kwargs):
    model = Protein_Clean_Seq_Anno_esmc(**kwargs)
    return model

class Protein_Clean_Seq_Anno_esmc_zero_shot(nn.Module):
    def __init__(self, seq_model=None, args=None):
        super().__init__()
        # Save args as class attribute
        self.args = args
        
        # Set default data type to float32
        torch.set_default_dtype(torch.float32)
        
        # Sequence encoder
        self.seq_encoder = seq_model
        
        # Ensure sequence encoder is always in evaluation mode
        self.seq_encoder.eval()
        for param in self.seq_encoder.parameters():
            param.requires_grad = False
        
        # Get dimension information
        self.seq_bert_dim = self.seq_encoder.embed.embedding_dim
        self.n_annotation = args.num_annotation if hasattr(args, 'num_annotation') else 384
        
        # Simplified feature extraction layer
        self.seq_proj = nn.Sequential(
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim),
            nn.LayerNorm(self.seq_bert_dim),
            nn.GELU()
        )
        
        # Annotation prediction layer
        self.anno_predictor = nn.Sequential(
            nn.Linear(self.seq_bert_dim, self.seq_bert_dim // 2),
            nn.LayerNorm(self.seq_bert_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.seq_bert_dim // 2, self.n_annotation)
        )
        
        # Initialize evaluation metrics
        self.metrics = {
            'accuracy': BinaryAccuracy(threshold=0.5),
            'precision': BinaryPrecision(threshold=0.5),
            'recall': BinaryRecall(threshold=0.5),
            'auroc': AUROC(task="binary"),
            'f1': BinaryF1Score(threshold=0.5)
        }
    
    def forward(self, protein_seq, anno=None, mode=None, epoch=None):
        """
        Zero-shot prediction of annotation
        
        Args:
            protein_seq: Protein sequence encoding, shape [batch_size, seq_len]
            anno: Optional real annotation, for evaluation metrics calculation
            
        Returns:
            Dictionary containing prediction results and evaluation metrics (if real annotation is provided)
        """
        device = protein_seq.device
        
        # Initialize return value dictionary
        output_dict = {}
        metrics_dict = {}
        
        # Ensure input is on correct device
        encoder_input_seq = protein_seq.to(device, dtype=torch.long, non_blocking=True)
        
        # Use ESM model to process sequence (no gradient computation)
        with torch.cuda.amp.autocast(enabled=True):
            try:
                with torch.no_grad():
                    output = self.seq_encoder(encoder_input_seq)
                    seq_hidden_embeds = output.embeddings
            except RuntimeError as e:
                print(f"Error in seq_encoder: {e}")
                raise e
        
        # Ensure feature tensors are on correct device and have correct type
        seq_hidden_embeds = seq_hidden_embeds.to(device, dtype=torch.float32)
        
        # Extract [CLS] feature
        cls_feat = seq_hidden_embeds[:, 0]  # [batch_size, seq_bert_dim]
        
        # Feature projection
        seq_feat = self.seq_proj(cls_feat)  # [batch_size, seq_bert_dim]
        
        # Predict annotation
        anno_logits = self.anno_predictor(seq_feat)  # [batch_size, n_annotation]
        anno_probs = torch.sigmoid(anno_logits)
        anno_pred = (anno_probs > 0.5).float()
        
        output_dict['anno_logits'] = anno_logits
        output_dict['anno_probs'] = anno_probs
        output_dict['anno_pred'] = anno_pred
        
        # If real annotation is provided, compute evaluation metrics
        if anno is not None:
            anno = anno.to(device, dtype=torch.float32)
            
            # Compute loss
            pred_loss = F.binary_cross_entropy_with_logits(anno_logits, anno)
            output_dict['pred_loss'] = pred_loss
            
            # Compute evaluation metrics
            for metric_name, metric in self.metrics.items():
                metrics_dict[f'anno_{metric_name}'] = torch.tensor(
                    metric(anno_probs.cpu(), anno.cpu()),
                    device=device
                )
            
            # Compute F-max
            try:
                fmax = 0.0
                for threshold in (t/20 for t in range(21)):
                    f1_score = BinaryF1Score(threshold=threshold)
                    fmax = max(fmax, f1_score(anno_probs.cpu(), anno.cpu()))
                metrics_dict['anno_fmax'] = torch.tensor(fmax, device=device)
            except Exception as e:
                print(f"Error calculating F-max: {e}")
            
            # Compute AUPRC
            try:
                from sklearn.metrics import average_precision_score
                auprc = average_precision_score(
                    anno.cpu().numpy(),
                    anno_probs.detach().cpu().numpy(),
                    average='micro'
                )
                metrics_dict['anno_auprc'] = torch.tensor(auprc, device=device)
            except ImportError:
                print("sklearn not found, skipping AUPRC calculation")
            except Exception as e:
                print(f"Error calculating AUPRC: {e}")
        
        # Merge all return values
        return {**output_dict, **metrics_dict}
    
    def get_anno_logits(self, protein_seq):
        """
        Get logits of annotation probabilities
        
        Args:
            protein_seq: Protein sequence encoding, shape [batch_size, seq_len]
            
        Returns:
            anno_logits: Logits of annotation probabilities, shape [batch_size, n_annotation]
        """
        device = protein_seq.device
        
        # Ensure input is on correct device
        encoder_input_seq = protein_seq.to(device, dtype=torch.long, non_blocking=True)
            
        # Use ESM model to process sequence (no gradient computation)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                output = self.seq_encoder(encoder_input_seq)
                seq_hidden_embeds = output.embeddings.to(device, dtype=torch.float32)
        
        # Extract [CLS] feature
        cls_feat = seq_hidden_embeds[:, 0]
        
        # Feature projection
        seq_feat = self.seq_proj(cls_feat)
        
        # Predict annotation
        anno_logits = self.anno_predictor(seq_feat)
        
        return anno_logits

def protein_clean_seq_anno_esmc_zero_shot(**kwargs):
    model = Protein_Clean_Seq_Anno_esmc_zero_shot(**kwargs)
    return model
