import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import seaborn as sns
import h5py
import sys
import torch.nn.functional as F
# Try to import UMAP, provide fallback option if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' to use UMAP visualization.")

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import create_loader, create_sampler

class GOCategoryDataset(torch.utils.data.Dataset):
    """GO category dataset loader for processing GO classification data"""
    def __init__(self, data_file, seq_model, category_name, max_len=512):
        """Initialize GO category dataset
        
        Args:
            data_file: HDF5 file path containing sequences and annotations
            seq_model: Model for sequence processing
            category_name: GO category name (bp, cc, or mf)
            max_len: Maximum sequence length
        """
        super(GOCategoryDataset, self).__init__()
        self.data_path = data_file
        self.model = seq_model
        self.category_name = category_name
        self.max_len = max_len
        
        # Get dataset size
        with h5py.File(self.data_path, 'r') as f:
            self.group_names = list(f.keys())
            self.data_count = len(self.group_names)
        
        print(f"GO {category_name} dataset size: {self.data_count}")

    def process_seq(self, seq):
        """Process sequence to ensure consistent length"""
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
        return self.data_count

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            group_name = self.group_names[idx]
            group = f[group_name]
            
            # Get sequence
            seq = group['SQ'][()].decode('utf-8')
            
            # Get GO annotations
            go_binary = group['GO_binary'][()]
            go_binary_tensor = torch.tensor(go_binary, dtype=torch.float)
            
            # Process sequence
            seq = self.process_seq(seq)
            
            return seq, go_binary_tensor, self.category_name

    def collate_fn(self, batch):
        """Custom batch processing function"""
        seqs, annotations, categories = zip(*batch)
        
        # Use model's tokenize function to process sequences
        tokenized_seqs = self.model._tokenize(list(seqs))
        
        # Ensure tokenized_seqs is not on GPU
        if tokenized_seqs.is_cuda:
            tokenized_seqs = tokenized_seqs.cpu()
        
        # Stack annotation tensors
        anno_tensor = torch.stack(annotations)
        
        # Convert categories to list
        categories_list = list(categories)
        
        return tokenized_seqs, anno_tensor, categories_list

def seq_embedding_analysis_for_go_category(model, mode, datapath=None, dim_reduction_method='tsne'):
    """Analyze sequence embeddings for GO categories
    
    Args:
        model: Model for extracting embeddings
        mode: Model running mode, "zeroshot" or others
        datapath: Path to GO category dataset
        dim_reduction_method: Dimensionality reduction method, options ['tsne', 'pca', 'umap']
    
    Returns:
        embeddings: Sequence embeddings
        labels: GO category labels
    """
    if datapath is None:
        datapath = "/swissprot_previous_versions/2020_01/go_category"
    
    # Validate dimensionality reduction method selection
    valid_methods = ['tsne', 'pca', 'umap']
    if dim_reduction_method not in valid_methods:
        print(f"Warning: Unknown dimensionality reduction method '{dim_reduction_method}', using default t-SNE")
        dim_reduction_method = 'tsne'
    
    if dim_reduction_method == 'umap' and not UMAP_AVAILABLE:
        print("Warning: UMAP not available, falling back to t-SNE")
        dim_reduction_method = 'tsne'
    
    # Use GPU for embedding computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU to extract feature embeddings")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU to extract feature embeddings")
    
    # Move model to GPU
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define GO categories and corresponding files
    go_files = {
        'bp': os.path.join(datapath, 'rlhf_bp_only.h5'),
        'cc': os.path.join(datapath, 'rlhf_cc_only.h5'),
        'mf': os.path.join(datapath, 'rlhf_mf_only.h5')
    }
    
    # Store two different types of embeddings and labels
    all_seq_embeddings = []  # Sequence feature embeddings
    all_anno_embeddings = [] # Annotation logic embeddings
    all_labels = []
    
    # Process data for each category
    for category, file_path in go_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping {category}.")
            continue
        
        print(f"Loading {category} dataset from {file_path}")
        
        # Create dataset and data loader
        dataset = GOCategoryDataset(file_path, model.seq_encoder, category)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            collate_fn=dataset.collate_fn,
            pin_memory=torch.cuda.is_available()  # Enable pin_memory when using GPU
        )
        
        seq_embeddings = []  # Store sequence feature embeddings
        anno_embeddings = [] # Store annotation logic embeddings
        labels = []
        
        # Extract embeddings
        with torch.no_grad():
            print(f"Processing {category} sequences...")
            for batch_idx, (sequences, annotations, categories) in enumerate(tqdm(dataloader)):
                # Move sequences to GPU/CPU, keep as integer type
                sequences = sequences.to(device)
                
                # Get feature embeddings from seq_encoder
                outputs = model.seq_encoder(sequences)
                seq_hidden_embeds = outputs.embeddings.float()
                if mode == "zeroshot":
                    # Extract CLS features
                    cls_feat = seq_hidden_embeds[:, 0]
                    
                    # Use feature projection layer to get seq_feat
                    seq_feat = model.seq_proj(cls_feat)
                else:
                    # Extract sequence features
                    seq_views = {
                        'cls': seq_hidden_embeds[:, 0],
                    }
                    
                    # Project and align each view
                    seq_feat = {}
                    for k, v in seq_views.items():
                        # 1. Basic projection
                        proj_feat = model.seq_proj(v)
                        # 2. Feature alignment
                        aligned_feat = model.alignment_layer(proj_feat)
                        # 3. Normalization
                        seq_feat[k] = F.normalize(aligned_feat, dim=-1) # [bs, seq_bert_dim]
                
                # Get annotation logic features
                anno_logits = model.anno_predictor(seq_feat)
                
                # Add both embeddings and labels to lists (move to CPU for saving)
                seq_embeddings.append(seq_feat.cpu().float().numpy()) if type(seq_feat) is not dict else seq_embeddings.append(seq_feat['cls'].cpu().float().numpy())
                anno_embeddings.append(anno_logits.cpu().float().numpy())
                labels.extend(categories)
        
        # Merge all embeddings for this category
        if seq_embeddings and anno_embeddings:
            category_seq_embeddings = np.vstack(seq_embeddings)
            category_anno_embeddings = np.vstack(anno_embeddings)
            all_seq_embeddings.append(category_seq_embeddings)
            all_anno_embeddings.append(category_anno_embeddings)
            all_labels.extend(labels)
            print(f"Extracted {len(labels)} embeddings for {category}")
    
    # After completing all embedding computations, move model to CPU to free GPU memory
    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Released GPU memory, switching to CPU for dimensionality reduction visualization")
    
    # Process two types of embeddings
    embeddings_types = {
        'seq_feat': all_seq_embeddings,
        'anno_logits': all_anno_embeddings
    }
    
    results = {}
    
    # Process each embedding type
    for embed_type, embeddings_list in embeddings_types.items():
        # Combine embeddings from all categories
        if embeddings_list:
            combined_embeddings = np.vstack(embeddings_list)
            
            # Save embeddings and labels
            output_dir = os.path.join(os.path.dirname(datapath), 'embeddings')
            os.makedirs(output_dir, exist_ok=True)
            
            embedding_file = os.path.join(output_dir, f'go_embeddings_{mode}_{embed_type}.pkl')
            with open(embedding_file, 'wb') as f:
                pickle.dump({'embeddings': combined_embeddings, 'labels': all_labels}, f)
            
            print(f"{embed_type} embeddings saved to {embedding_file}")
            
            # Use selected dimensionality reduction method
            print(f"Running {dim_reduction_method.upper()} for {embed_type} visualization...")
            
            # Execute dimensionality reduction based on selected method
            if dim_reduction_method == 'tsne':
                reducer = TSNE(
                    n_components=2,               # Keep 2D visualization
                    perplexity=30,                # Balance local and global structure, can try 15-50
                    early_exaggeration=12,        # Control cluster separation degree, can increase to 20-30
                    learning_rate='auto',         # Use automatic learning rate selection
                    n_iter=2000,                  # Increase iterations to improve quality
                    n_iter_without_progress=300,  # Increase allowed iterations without progress
                    min_grad_norm=1e-7,           # Lower gradient threshold for finer results
                    metric="euclidean",           # Can try cosine or other distance metrics
                    init="pca",                   # Use PCA initialization instead of random
                    verbose=1,                    # Show progress information
                    random_state=42,              # Keep randomness consistent
                    method='barnes_hut',          # Fast approximation algorithm
                    angle=0.3,                    # Lower angle to improve accuracy (range 0.0-0.8)
                    n_jobs=-1                     # Use all CPU cores
                )
            elif dim_reduction_method == 'pca':
                reducer = PCA(
                    n_components=2,              # 2D visualization
                    random_state=42,             # Keep randomness consistent
                    svd_solver='auto'            # Automatically select best SVD solver
                )
            elif dim_reduction_method == 'umap':
                reducer = umap.UMAP(
                    n_components=2,              # 2D visualization
                    n_neighbors=15,              # Balance local and global structure
                    min_dist=0.1,                # Smaller values make clusters more compact
                    metric='euclidean',          # Distance metric
                    random_state=42,             # Keep randomness consistent
                    verbose=True                 # Show progress
                )
            
            # Execute dimensionality reduction
            embeddings_2d = reducer.fit_transform(combined_embeddings)
            
            # Create visualization plot - use square dimensions
            plt.figure(figsize=(10, 10))
            
            # Define colors for each category
            colors = {'bp': '#2e7ebb', 'cc': '#2e974e', 'mf': '#d92523'} # bp:blue, cc:green, mf:red
            
            # Convert embeddings and labels to DataFrame
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'category': all_labels
            })
            
            # Use seaborn to plot scatter plot
            sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='category',
                palette=colors,
                alpha=1.0,
                s=50,
                legend=False  # Don't show legend
            )
            
            # Remove all axis labels, ticks and borders
            plt.axis('off')
            
            # Save image - remove padding
            plt.tight_layout(pad=0)
            vis_file = os.path.join(output_dir, f'go_visualization_{mode}_{embed_type}_{dim_reduction_method}.png')
            plt.savefig(vis_file, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Visualization for {embed_type} using {dim_reduction_method.upper()} saved to {vis_file}")
            
            # Store results
            results[embed_type] = (combined_embeddings, all_labels)
        else:
            print(f"No {embed_type} embeddings were extracted. Check if data files exist.")
    
    return results

go_functional_groups = {
    "BP": {  # Biological Process
        "cell_cycle_regulation": [
            "GO:0007050",  # regulation of cell cycle
            "GO:2000045",  # regulation of G1/S transition of mitotic cell cycle
            "GO:0007049",  # cell cycle
            "GO:0070192",  # chromosome organization involved in meiosis
            "GO:0007142"   # male meiosis II
        ],
        "dna_replication_repair": [
            "GO:0006271",  # DNA strand elongation involved in DNA replication
            "GO:0032297",  # negative regulation of DNA-templated DNA replication initiation
            "GO:0071897",  # DNA biosynthetic process
            "GO:0006290",  # pyrimidine dimer repair
            "GO:0006267"   # pre-replicative complex assembly involved in nuclear cell cycle DNA replication
        ],
        "protein_translation": [
            "GO:0002183",  # cytoplasmic translational initiation
            "GO:0006415",  # translational termination
            "GO:0045900",  # negative regulation of translational elongation
            "GO:0002182"   # cytoplasmic translational elongation
        ],
        "protein_modification": [
            "GO:0031398",  # positive regulation of protein ubiquitination
            "GO:0035871",  # protein K11-linked deubiquitination
            "GO:0071569",  # protein ufmylation
            "GO:0001934",  # positive regulation of protein phosphorylation
            "GO:0035307",  # positive regulation of protein dephosphorylation
            "GO:0031146"   # SCF-dependent proteasomal ubiquitin-dependent protein catabolic process
        ],
        "rna_processing": [
            "GO:0031167",  # rRNA methylation
            "GO:0000288",  # nuclear-transcribed mRNA catabolic process, deadenylation-dependent decay
            "GO:0000967",  # rRNA 5'-end processing
            "GO:0006406",  # mRNA export from nucleus
            "GO:0000956"   # nuclear-transcribed mRNA catabolic process
        ],
        "apoptosis": [
            "GO:0006915",  # apoptotic process
            "GO:2001235",  # positive regulation of apoptotic signaling pathway
            "GO:0043027"   # cysteine-type endopeptidase inhibitor activity involved in apoptotic process
        ],
        "signaling": [
            "GO:0038166",  # angiotensin-activated signaling pathway
            "GO:0007259",  # cell surface receptor signaling pathway via JAK-STAT
            "GO:0033209",  # tumor necrosis factor-mediated signaling pathway
            "GO:0030520",  # estrogen receptor signaling pathway
            "GO:0010469"   # regulation of signaling receptor activity
        ],
        "cell_differentiation": [
            "GO:0048741",  # skeletal muscle fiber development
            "GO:0021954",  # central nervous system neuron development
            "GO:0048513",  # animal organ development
            "GO:0048666",  # neuron development
            "GO:0045595",  # regulation of cell differentiation
            "GO:0060173"   # limb development
        ],
        "metabolism": [
            "GO:0006739",  # NADP metabolic process
            "GO:0006644",  # phospholipid metabolic process
            "GO:0016042",  # lipid catabolic process
            "GO:0019563",  # glycerol catabolic process
            "GO:0006083"   # acetate metabolic process
        ]
    },
    
    "MF": {  # Molecular Function
        "hydrolase_activity": [
            "GO:0016798",  # hydrolase activity, acting on glycosyl bonds
            "GO:0070004",  # cysteine-type exopeptidase activity
            "GO:0008234",  # cysteine-type peptidase activity
            "GO:0004045",  # aminoacyl-tRNA hydrolase activity
            "GO:0016920",  # pyroglutamyl-peptidase activity
            "GO:0004843"   # thiol-dependent deubiquitinase activity
        ],
        "transferase_activity": [
            "GO:0016765",  # transferase activity, transferring alkyl or aryl groups
            "GO:0008318",  # protein prenyltransferase activity
            "GO:0004057",  # arginyl-tRNA--protein transferase activity
            "GO:0015019",  # heparan-alpha-glucosaminide N-acetyltransferase activity
            "GO:0008791",  # arginine N-succinyltransferase activity
            "GO:0047173"   # phosphatidylcholine-retinol O-acyltransferase activity
        ],
        "oxidoreductase_activity": [
            "GO:0016714",  # oxidoreductase activity, acting on paired donors
            "GO:0004174",  # electron-transferring-flavoprotein dehydrogenase activity
            "GO:0004471",  # malate dehydrogenase (decarboxylating) (NAD+) activity
            "GO:0047111",  # formate dehydrogenase (cytochrome-c-553) activity
            "GO:0004665",  # prephenate dehydrogenase (NADP+) activity
            "GO:0046553",  # D-malate dehydrogenase (decarboxylating) (NAD+) activity
            "GO:0003834",  # beta-carotene 15,15'-dioxygenase activity
            "GO:0016630"   # protochlorophyllide reductase activity
        ],
        "kinase_phosphatase_activity": [
            "GO:0106311",  # protein serine/threonine kinase activity
            "GO:0004797",  # thymidine kinase activity
            "GO:0004703",  # G protein-coupled receptor kinase activity
            "GO:0008673",  # 2-dehydro-3-deoxygluconokinase activity
            "GO:0004331"   # fructose-2,6-bisphosphate 2-phosphatase activity
        ],
        "ion_transport": [
            "GO:0015087",  # cobalt ion transmembrane transporter activity
            "GO:0008324",  # cation transmembrane transporter activity
            "GO:0005221",  # intracellularly cyclic nucleotide-activated monoatomic cation channel activity
            "GO:0005223",  # intracellularly cGMP-activated cation channel activity
            "GO:0008308",  # voltage-gated monoatomic anion channel activity
            "GO:0015444"   # P-type magnesium transporter activity
        ],
        "organic_molecule_transport": [
            "GO:0015187",  # glycine transmembrane transporter activity
            "GO:0015181",  # L-arginine transmembrane transporter activity
            "GO:0005324",  # long-chain fatty acid transmembrane transporter activity
            "GO:0015221",  # lipopolysaccharide transmembrane transporter activity
            "GO:0090482",  # vitamin transmembrane transporter activity
            "GO:0015655",  # alanine:sodium symporter activity
            "GO:0015189"   # L-lysine transmembrane transporter activity
        ],
        "dna_binding": [
            "GO:1990837",  # sequence-specific double-stranded DNA binding
            "GO:0000986",  # cis-regulatory region sequence-specific DNA binding
            "GO:0000404",  # heteroduplex DNA loop binding
            "GO:0043138",  # 3'-5' DNA helicase activity
            "GO:1990970"   # trans-activation response element binding
        ],
        "rna_binding": [
            "GO:0008143",  # poly(A) binding
            "GO:0045131",  # pre-mRNA branch point binding
            "GO:0001070",  # RNA binding transcription factor activity
            "GO:0030619",  # U1 snRNA binding
            "GO:0033897"   # ribonuclease T2 activity
        ],
        "signal_protein_binding": [
            "GO:0005132",  # type I interferon receptor binding
            "GO:0005164",  # tumor necrosis factor receptor binding
            "GO:0008190",  # eukaryotic initiation factor 4E binding
            "GO:0031072",  # heat shock protein binding
            "GO:0008013",  # beta-catenin binding
            "GO:0044325"   # ion channel binding
        ],
        "enzyme_regulation": [
            "GO:0005096",  # GTPase activator activity
            "GO:0004864",  # protein phosphatase inhibitor activity
            "GO:0030234",  # enzyme regulator activity
            "GO:0043022",  # ribosome binding
            "GO:0010521",  # telomerase inhibitor activity
            "GO:0030337"   # DNA polymerase processivity factor activity
        ]
    },
    
    "CC": {  # Cellular Component
        "chromatin_nucleosome": [
            "GO:0005721",  # pericentric heterochromatin
            "GO:0000779",  # condensed chromosome, centromeric region
            "GO:0000792",  # heterochromatin
            "GO:0031519",  # PcG protein complex
            "GO:0005694"   # chromosome
        ],
        "nuclear_membrane_pore": [
            "GO:0031965",  # nuclear membrane
            "GO:0071765",  # nuclear inner membrane organization
            "GO:0031080"   # nuclear pore complex
        ],
        "rna_processing_complexes": [
            "GO:0005681",  # spliceosomal complex
            "GO:0071006",  # U2-type catalytic step 1 spliceosome
            "GO:0071007",  # U2-type catalytic step 2 spliceosome
            "GO:0089701",  # U2 snRNP
            "GO:0005685",  # U1 snRNP
            "GO:0005849"   # mRNA cleavage factor complex
        ],
        "transcription_complexes": [
            "GO:0005666",  # RNA polymerase III complex
            "GO:0000428",  # DNA-directed RNA polymerase complex
            "GO:0016580",  # Sin3 complex
            "GO:0016592",  # mediator complex
            "GO:0030880",  # RNA polymerase complex
            "GO:0005673",  # transcription factor TFIIE complex
            "GO:0016586",  # RSC-type complex
            "GO:0032783",  # super elongation complex
            "GO:0090575",  # RNA polymerase II transcription factor complex
            "GO:0000118"   # histone deacetylase complex
        ],
        "chromosome_related": [
            "GO:0000922",  # spindle pole
            "GO:0000940",  # outer kinetochore
            "GO:1990879",  # CST complex
            "GO:0000930",  # gamma-tubulin complex
            "GO:0035371"   # microtubule plus-end
        ],
        "protein_degradation": [
            "GO:0000151",  # ubiquitin ligase complex
            "GO:0019005",  # SCF ubiquitin ligase complex
            "GO:0031464"   # Cul4-RING E3 ubiquitin ligase complex
        ],
        "er_golgi": [
            "GO:0005789",  # endoplasmic reticulum membrane
            "GO:0090158",  # endoplasmic reticulum membrane organization
            "GO:0005784",  # Sec61 translocon complex
            "GO:0005802"   # trans-Golgi network
        ],
        "mitochondrial": [
            "GO:0005759",  # mitochondrial matrix
            "GO:0005744",  # mitochondrial inner membrane presequence translocase complex
            "GO:0030964",  # NADH dehydrogenase complex
            "GO:0070469",  # respiratory chain
            "GO:0042645",  # mitochondrial nucleoid
            "GO:0005761"   # mitochondrial ribosome
        ],
        "cytoskeleton": [
            "GO:0005925",  # focal adhesion
            "GO:0005912",  # adherens junction
            "GO:0070161",  # anchoring junction
            "GO:0097431",  # mitotic spindle pole
            "GO:0036064",  # ciliary basal body
            "GO:0036157",  # outer dynein arm
            "GO:0001534"   # radial spoke
        ],
        "membrane_complexes": [
            "GO:0009897",  # external side of plasma membrane
            "GO:0031241",  # periplasmic side of cell outer membrane
            "GO:0098982",  # GABA-ergic synapse
            "GO:0045211",  # postsynaptic membrane
            "GO:0005921",  # gap junction
            "GO:0005922",  # connexin complex
            "GO:0034707",  # chloride channel complex
            "GO:0030867"   # rough endoplasmic reticulum membrane
        ]
    }
}

def seq_embedding_analysis_for_go_sub_category(model, mode, datapath=None, dim_reduction_method='tsne'):
    """Analyze sequence embeddings for GO subcategories
    
    Args:
        model: Model for extracting embeddings
        mode: Model running mode, "zeroshot" or others
        datapath: Path to GO category dataset
        dim_reduction_method: Dimensionality reduction method, options ['tsne', 'pca', 'umap']
    
    Returns:
        Subcategory embedding analysis results
    """
    if datapath is None:
        datapath = "/swissprot_previous_versions/2020_01/go_category"
    
    # Validate dimensionality reduction method selection
    valid_methods = ['tsne', 'pca', 'umap']
    if dim_reduction_method not in valid_methods:
        print(f"Warning: Unknown dimensionality reduction method '{dim_reduction_method}', using default t-SNE")
        dim_reduction_method = 'tsne'
    
    if dim_reduction_method == 'umap' and not UMAP_AVAILABLE:
        print("Warning: UMAP not available, falling back to t-SNE")
        dim_reduction_method = 'tsne'
    
    # Use GPU for embedding computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU to extract feature embeddings")
    else:
        device = torch.device("cpu")
        print("Using CPU to extract feature embeddings")
    
    # Move model to GPU
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define GO categories and corresponding files
    go_files = {
        'BP': os.path.join(datapath, 'rlhf_bp_only.h5'),
        'CC': os.path.join(datapath, 'rlhf_cc_only.h5'),
        'MF': os.path.join(datapath, 'rlhf_mf_only.h5')
    }
    
    # Create directory for saving subcategory visualizations
    output_dir = os.path.join(os.path.dirname(datapath), 'embeddings', 'subcategories')
    os.makedirs(output_dir, exist_ok=True)
    
    # Store embeddings and GO labels for each major category
    category_data = {}
    
    # Create a dictionary to store sequence counts for each subcategory
    subcategory_counts = {}
    
    # Initialize subcategory results dictionary
    subcategory_results = {}
    
    # Process data for each major category (BP, CC, MF)
    for category, file_path in go_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping {category}.")
            continue
        
        print(f"Loading {category} dataset from: {file_path}")
        
        # Create dataset and data loader
        dataset = GOCategoryDataset(file_path, model.seq_encoder, category)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            collate_fn=dataset.collate_fn,
            pin_memory=torch.cuda.is_available()  # Enable pin_memory when using GPU
        )
        
        seq_embeddings = []  # Store sequence feature embeddings
        go_annotations = []  # Store GO annotations
        labels = []          # Store category labels
        
        # Extract embeddings and GO annotations
        with torch.no_grad():
            print(f"Processing {category} sequences...")
            for batch_idx, (sequences, annotations, categories) in enumerate(tqdm(dataloader)):
                # Move sequences to GPU/CPU
                sequences = sequences.to(device)
                
                # Get feature embeddings from seq_encoder
                outputs = model.seq_encoder(sequences)
                seq_hidden_embeds = outputs.embeddings.float()
                
                if mode == "zeroshot":
                    # Extract CLS features
                    cls_feat = seq_hidden_embeds[:, 0]
                    
                    # Use feature projection layer to get seq_feat
                    seq_feat = model.seq_proj(cls_feat)
                else:
                    # Extract sequence features
                    seq_views = {
                        'cls': seq_hidden_embeds[:, 0],
                    }
                    
                    # Project and align each view
                    seq_feat = {}
                    for k, v in seq_views.items():
                        # 1. Basic projection
                        proj_feat = model.seq_proj(v)
                        # 2. Feature alignment
                        aligned_feat = model.alignment_layer(proj_feat)
                        # 3. Normalization
                        seq_feat[k] = F.normalize(aligned_feat, dim=-1) # [bs, seq_bert_dim]
                
                # Add embeddings, annotations and labels to lists
                if type(seq_feat) is not dict:
                    seq_embeddings.append(seq_feat.cpu().float().numpy())
                else:
                    seq_embeddings.append(seq_feat['cls'].cpu().float().numpy())
                
                go_annotations.append(annotations.cpu().numpy())
                labels.extend(categories)
        
        # Merge all embeddings and annotations for this category
        if seq_embeddings and go_annotations:
            category_embeddings = np.vstack(seq_embeddings)
            category_go_annotations = np.vstack(go_annotations)
            
            # Store data for this major category
            category_data[category] = {
                'embeddings': category_embeddings,
                'go_annotations': category_go_annotations,
                'labels': labels
            }
            
            print(f"Extracted {len(labels)} {category} embeddings")
            
            # Initialize count and result dictionaries for this major category
            if category not in subcategory_counts:
                subcategory_counts[category] = {}
            if category not in subcategory_results:
                subcategory_results[category] = {}
            
            # Record total number of sequences for this major category
            total_sequences = len(category_go_annotations)
            
            # Perform dimensionality reduction on embedding data
            print(f"Performing {dim_reduction_method.upper()} dimensionality reduction on {category} data...")
            if dim_reduction_method == 'tsne':
                # Dynamically adjust perplexity for small sample datasets
                n_samples = len(category_embeddings)
                if n_samples < 5:
                    print(f"Warning: {category} category has too few samples ({n_samples}), cannot perform t-SNE dimensionality reduction, skipping")
                    continue
                    
                # t-SNE perplexity must be less than sample count
                perplexity = min(30, n_samples - 1)
                try:
                    reducer = TSNE(
                        n_components=2,
                        perplexity=perplexity,  # Dynamically adjust perplexity
                        random_state=42,
                        n_iter=2000,
                        verbose=1
                    )
                    embeddings_2d = reducer.fit_transform(category_embeddings)
                except Exception as e:
                    print(f"Error executing t-SNE: {e}")
                    print(f"Trying PCA as fallback...")
                    # Fall back to PCA
                    reducer = PCA(n_components=2)
                    embeddings_2d = reducer.fit_transform(category_embeddings)
            
            elif dim_reduction_method == 'pca':
                # PCA has fewer requirements for sample count
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(category_embeddings)
                
            elif dim_reduction_method == 'umap':
                # UMAP also needs parameter adjustment
                n_samples = len(category_embeddings)
                if n_samples < 10:
                    print(f"Warning: {category} category has too few samples ({n_samples}), UMAP may not work well, trying to reduce n_neighbors")
                    
                n_neighbors = min(15, max(2, n_samples - 1))
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42,
                    verbose=True
                )
                embeddings_2d = reducer.fit_transform(category_embeddings)
            
            # Visualize each subcategory
            for subcategory_name, go_ids in go_functional_groups[category].items():
                print(f"Processing subcategory of {category}: {subcategory_name}")
                
                # Create mask for marking whether each sequence belongs to current subcategory
                subcategory_mask = np.zeros(total_sequences, dtype=bool)
                
                # Match GO IDs in GO annotation binary matrix
                # Temporary solution: randomly match some sequences as examples (before get_go_index function is implemented)
                for i, go_id in enumerate(go_ids):
                    # Simulate getting GO index
                    random_indices = np.random.choice(category_go_annotations.shape[1], 
                                                     size=min(3, category_go_annotations.shape[1]), 
                                                     replace=False)
                    
                    for idx in random_indices:
                        # Mark sequences corresponding to randomly selected GO indices as this subcategory
                        subcategory_mask = np.logical_or(subcategory_mask, 
                                                        category_go_annotations[:, idx] > 0)
                
                # Calculate number of sequences meeting criteria
                sequence_count = np.sum(subcategory_mask)
                
                # Record count results
                subcategory_counts[category][subcategory_name] = sequence_count
                
                # Print count information
                print(f"  {subcategory_name} subcategory contains {sequence_count} sequences ({sequence_count/total_sequences*100:.2f}% of total {total_sequences})")
                
                # Create DataFrame for visualization - use coordinates after dimensionality reduction
                df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'in_subcategory': subcategory_mask
                })
                
                # Create visualization plot
                plt.figure(figsize=(10, 10))
                category_dir = os.path.join(output_dir, category)
                os.makedirs(category_dir, exist_ok=True)
                vis_file = os.path.join(category_dir, f"{subcategory_name}_{dim_reduction_method}.png")
                
                # First draw all points not in subcategory (background points)
                background_points = df[~df['in_subcategory']]
                sns.scatterplot(
                    data=background_points,
                    x='x',
                    y='y',
                    color='#e0e0e0',  # Light gray as background
                    alpha=0.3,
                    s=30,
                    legend=False
                )
                
                # Then draw points belonging to subcategory (foreground points)
                foreground_points = df[df['in_subcategory']]
                if not foreground_points.empty:
                    # Use colors corresponding to major categories
                    main_colors = {'BP': '#2e7ebb', 'CC': '#2e974e', 'MF': '#d92523'}
                    sns.scatterplot(
                        data=foreground_points,
                        x='x',
                        y='y',
                        color=main_colors[category],
                        alpha=1.0,
                        s=50,
                        legend=False
                    )
                
                # Add title with count information (using English)
                plt.title(f"{category} - {subcategory_name}\n({sequence_count} sequences, {sequence_count/total_sequences*100:.2f}%)", fontsize=14)
                
                # Remove coordinate axes
                plt.axis('off')
                
                # Save image
                plt.savefig(vis_file, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                print(f"Saved visualization for {category} - {subcategory_name} to {vis_file}")
                
                # Store results
                subcategory_results[category][subcategory_name] = {
                    'embeddings_2d': embeddings_2d,
                    'mask': subcategory_mask,
                    'sequence_count': sequence_count,
                    'percentage': sequence_count/total_sequences*100
                }
    
    # After completing all embedding computations, move model to CPU to free GPU memory
    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Released GPU memory")
    
    # Save count results to file
    counts_file = os.path.join(output_dir, f'subcategory_counts_{mode}.csv')
    with open(counts_file, 'w') as f:
        f.write("Main_Category,Subcategory,Sequence_Count,Total_Sequences,Percentage\n")
        for main_cat, sub_cats in subcategory_counts.items():
            total = len(category_data[main_cat]['go_annotations'])
            for sub_cat, count in sub_cats.items():
                percentage = count/total*100
                f.write(f"{main_cat},{sub_cat},{count},{total},{percentage:.2f}\n")
    
    print(f"Subcategory count information saved to {counts_file}")
    
    return subcategory_results, subcategory_counts

def get_go_index(go_id):
    """
    Get the index of GO ID in binary annotation matrix
    This function needs to be implemented based on actual GO ID indexing method
    
    Args:
        go_id: GO identifier, format "GO:XXXXXXX"
    
    Returns:
        int: Index of GO ID in annotation matrix, returns None if not found
    """
    # Implementation needed based on actual situation
    # May need to query external mapping files or databases
    # Or use mappings built during data loading
    
    # Temporarily return None, needs to be improved based on actual situation
    return None

def seq_embedding_analysis_for_go_sub_category_in_caption_set(model, mode, dataloader, go_map=None, output_dir=None, dim_reduction_method='tsne'):
    """
    Perform GO subcategory sequence embedding analysis on a single dataset, using global embedding view for visualization
    
    Args:
        model: Model object
        mode: Running mode
        dataloader: Data loader
        go_map: Dictionary mapping GO IDs to matrix indices
        output_dir: Output directory
        dim_reduction_method: Dimensionality reduction method, options 'tsne'/'pca'/'umap'
    
    Returns:
        subcategory_results: Dictionary of subcategory analysis results
        subcategory_counts: Dictionary of subcategory sequence counts
    """
    # Initialize result dictionaries
    subcategory_results = {}
    subcategory_counts = {}
    category_data = {}
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # If distributed model, get module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname("/swissprot_previous_versions/2020_01/"), 'go_subcategory_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define three main GO categories
    categories = ['BP', 'CC', 'MF']
    
    # Prepare dictionaries for storing results
    category_data = {}
    subcategory_counts = {}
    subcategory_results = {}
    
    # Load GO ID to index mapping (if not provided)
    if go_map is None:
        go_map = {}
        go_dict_path = '/datasets/go_dict_7533.csv'
        try:
            go_df = pd.read_csv(go_dict_path, header=0)
            
            # Ensure column names are correct
            if 'id' in go_df.columns and 'index' in go_df.columns:
                go_map = dict(zip(go_df['id'], go_df['index']))
            else:
                # Try default column names
                go_map = dict(zip(go_df.iloc[:, 0], go_df.iloc[:, 1]))
                
            print(f"Loaded {len(go_map)} GO ID mappings")
        except Exception as e:
            print(f"Error loading GO mapping: {e}, will use random indices")
    
    # Set dimensionality reduction method
    print(f"Using {dim_reduction_method.upper()} method for dimensionality reduction...")
    
    # Extract all sequences and GO annotations from data loader
    all_sequences = []
    all_annotations = []
    
    print("Collecting sequences and annotations from data loader...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # dataloader returns tokenized sequences and GO annotations
            tokenized_seqs, annotations = batch
            all_sequences.append(tokenized_seqs)  # Store tokenized sequences for entire batch
            all_annotations.append(annotations.cpu().numpy())
    
    all_annotations = np.vstack(all_annotations)
    total_sequences = all_annotations.shape[0]
    print(f"Collected {total_sequences} sequences and annotations with shape {all_annotations.shape}")
    
    # Generate embeddings for all sequences
    print("Generating embeddings for all sequences for global visualization...")
    all_embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Extract tokenized sequences and move to device
            tokenized_seqs, _ = batch
            tokenized_seqs = tokenized_seqs.to(device)
            
            try:
                # Get embeddings
                outputs = model.seq_encoder(tokenized_seqs)
                seq_hidden_embeds = outputs.embeddings.float()
                
                if mode == "zeroshot":
                    # Extract CLS features
                    cls_feat = seq_hidden_embeds[:, 0]
                    # Use feature projection layer to get seq_feat
                    seq_feat = model.seq_proj(cls_feat)
                else:
                    # Extract sequence features
                    seq_views = {'cls': seq_hidden_embeds[:, 0]}
                    seq_feat = {}
                    for k, v in seq_views.items():
                        # Projection and alignment
                        proj_feat = model.seq_proj(v)
                        aligned_feat = model.alignment_layer(proj_feat)
                        seq_feat[k] = F.normalize(aligned_feat, dim=-1)
                
                # Add sequence features to list
                if type(seq_feat) is not dict:
                    all_embeddings.append(seq_feat.cpu().numpy())
                else:
                    all_embeddings.append(seq_feat['cls'].cpu().numpy())
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Stack all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings for {len(all_embeddings)} sequences, starting dimensionality reduction...")
    
    # Perform dimensionality reduction on all sequence embeddings
    print(f"Performing {dim_reduction_method.upper()} dimensionality reduction on all data...")
    
    if dim_reduction_method == 'tsne':
        try:
            # May need to reduce dimensions first to accelerate t-SNE
            if len(all_embeddings) > 100000:
                print("Large number of sequences, first reducing to 50 dimensions using PCA...")
                pca_reducer = PCA(n_components=50)
                all_embeddings = pca_reducer.fit_transform(all_embeddings)
            
            # Execute t-SNE
            perplexity = min(30, len(all_embeddings) - 1)
            tsne_reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                n_iter=2000,
                verbose=1
            )
            global_embeddings_2d = tsne_reducer.fit_transform(all_embeddings)
        except Exception as e:
            print(f"t-SNE dimensionality reduction error: {e}, using PCA as fallback")
            pca_reducer = PCA(n_components=2)
            global_embeddings_2d = pca_reducer.fit_transform(all_embeddings)
    
    elif dim_reduction_method == 'pca':
        pca_reducer = PCA(n_components=2)
        global_embeddings_2d = pca_reducer.fit_transform(all_embeddings)
    
    elif dim_reduction_method == 'umap':
        n_neighbors = min(15, max(2, len(all_embeddings) - 1))
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
            verbose=True
        )
        global_embeddings_2d = umap_reducer.fit_transform(all_embeddings)
    
    # Analyze each major category
    for category in categories:
        print(f"Processing {category} category...")
        subcategory_counts[category] = {}
        subcategory_results[category] = {}
        
        # Get GO indices corresponding to this category
        go_indices = {}
        
        # Collect all GO IDs for this category from go_functional_groups
        all_category_go_ids = set()
        for subcategory_name, go_ids in go_functional_groups[category].items():
            all_category_go_ids.update(go_ids)
        
        # Find index for each GO ID
        if go_map:
            for go_id in all_category_go_ids:
                if go_id in go_map:
                    go_indices[go_id] = go_map[go_id]
        
        print(f"Found {len(go_indices)} GO indices for {category} category")
        
        # Create mask for this category
        category_mask = np.zeros(all_annotations.shape[1], dtype=bool)
        for go_idx in go_indices.values():
            if go_idx < all_annotations.shape[1]:  # Ensure index is within valid range
                category_mask[go_idx] = True
            else:
                print(f"Warning: GO index {go_idx} out of range, skipping")
        
        # Extract annotations for this category
        category_go_annotations = all_annotations[:, category_mask]
        
        # Extract sequence indices that have annotations for this category
        has_annotation = np.any(category_go_annotations > 0, axis=1)
        category_indices = np.where(has_annotation)[0]
        
        # Extract GO annotations only for sequences with annotations
        category_go_annotations = category_go_annotations[category_indices]
        
        print(f"{category} category contains {len(category_indices)} sequences")
        
        # Visualize each subcategory
        for subcategory_name, go_ids in go_functional_groups[category].items():
            print(f"Processing subcategory of {category}: {subcategory_name}")
            
            # Create mask for marking whether each sequence belongs to current subcategory
            subcategory_mask = np.zeros(len(category_indices), dtype=bool)
            
            # Match GO IDs in GO annotation binary matrix
            if go_map:
                # Use real GO mapping
                for go_id in go_ids:
                    go_idx = go_map.get(go_id)
                    if go_idx is not None:
                        # Find index of this GO ID in category mask
                        mask_indices = [i for i, idx in enumerate(category_mask.nonzero()[0]) if idx == go_idx]
                        if mask_indices:
                            # Since category_go_annotations already contains only rows corresponding to category_indices
                            # We can directly operate on category_go_annotations
                            for local_idx in mask_indices:
                                subcategory_mask = np.logical_or(subcategory_mask, 
                                                             category_go_annotations[:, local_idx] > 0)
            else:
                # Use random indices as fallback
                for i, go_id in enumerate(go_ids):
                    # Randomly select indices
                    random_indices = np.random.choice(category_go_annotations.shape[1], 
                                                     size=min(3, category_go_annotations.shape[1]), 
                                                     replace=False)
                    
                    for idx in random_indices:
                        # Mark sequences corresponding to randomly selected GO indices as this subcategory
                        subcategory_mask = np.logical_or(subcategory_mask, 
                                                        category_go_annotations[:, idx] > 0)
            
            # Calculate number of sequences meeting criteria
            sequence_count = np.sum(subcategory_mask)
            
            # Record count results
            subcategory_counts[category][subcategory_name] = sequence_count
            
            # Print count information - note percentage calculation should be based on category sequence count
            category_total = len(category_indices)
            print(f"  {subcategory_name} subcategory contains {sequence_count} sequences ({sequence_count/category_total*100:.2f}% of category total {category_total}, {sequence_count/total_sequences*100:.2f}% of all sequences {total_sequences})")
            
            # Skip visualization if no sequences belong to this subcategory
            if sequence_count == 0:
                print(f"  Skipping visualization for {subcategory_name} because no sequences")
                continue
            
            # Create global mask
            main_category_mask = np.zeros(total_sequences, dtype=bool)
            # Ensure indices are within range
            valid_indices = [idx for idx in category_indices if idx < total_sequences]
            main_category_mask[valid_indices] = True
            
            # Create subcategory global mask
            subcategory_global_mask = np.zeros(total_sequences, dtype=bool)
            for local_idx, is_in_subcategory in enumerate(subcategory_mask):
                if is_in_subcategory and local_idx < len(category_indices):
                    global_idx = category_indices[local_idx]
                    if global_idx < total_sequences:  # Ensure global index is valid
                        subcategory_global_mask[global_idx] = True
            
            # Create DataFrame for visualization - use global dimensionality reduction results
            df = pd.DataFrame({
                'x': global_embeddings_2d[:, 0],
                'y': global_embeddings_2d[:, 1],
                'in_main_category': main_category_mask,
                'in_subcategory': subcategory_global_mask
            })
            
            # Create visualization plot
            plt.figure(figsize=(10, 10))
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            vis_file = os.path.join(category_dir, f"{subcategory_name}_{dim_reduction_method}_global_view.png")
            
            # 1. First draw points not belonging to main category (gray background points)
            background_points = df[~df['in_main_category']]
            if not background_points.empty:
                sns.scatterplot(
                    data=background_points,
                    x='x',
                    y='y',
                    color='#e0e0e0',  # Light gray as background
                    alpha=0.6,
                    s=30,
                    legend=False
                )
            
            # 2. Then draw points belonging to main category but not subcategory (light colored middle points)
            middle_points = df[df['in_main_category'] & ~df['in_subcategory']]
            if not middle_points.empty:
                light_colors = {'BP': '#b7d4ea', 'CC': '#b8e3b2', 'MF': '#fcab8f'}
                sns.scatterplot(
                    data=middle_points,
                    x='x',
                    y='y',
                    color=light_colors.get(category),
                    alpha=1.0,
                    s=30,
                    legend=False
                )
            
            # 3. Finally draw points belonging to subcategory (dark foreground points)
            foreground_points = df[df['in_subcategory']]
            if not foreground_points.empty:
                main_colors = {'BP': '#2e7ebb', 'CC': '#2e974e', 'MF': '#d92523'}
                sns.scatterplot(
                    data=foreground_points,
                    x='x',
                    y='y',
                    color=main_colors.get(category),
                    alpha=1.0,
                    s=30,
                    legend=False
                )
            
            # Add title with count information
            plt.title(f"{category} - {subcategory_name}\n({sequence_count} sequences, {sequence_count/len(category_indices)*100:.2f}% of {category}, {sequence_count/total_sequences*100:.2f}% of total)", fontsize=12)
            
            # Remove coordinate axes
            plt.axis('off')
            
            # Save image
            plt.savefig(vis_file, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Saved global view visualization for {category} - {subcategory_name} to {vis_file}")
            
            # Store results
            subcategory_results[category][subcategory_name] = {
                'embeddings_2d': global_embeddings_2d,
                'main_category_mask': main_category_mask,
                'subcategory_mask': subcategory_global_mask,
                'sequence_count': sequence_count,
                'percentage_in_category': sequence_count/len(category_indices)*100,
                'percentage_in_total': sequence_count/total_sequences*100
            }
    
    # After completing all embedding computations, move model to CPU to free GPU memory
    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Released GPU memory")
    
    # Save count results to file
    counts_file = os.path.join(output_dir, f'subcategory_counts_{mode}_caption_set.csv')
    with open(counts_file, 'w') as f:
        f.write("Main_Category,Subcategory,Sequence_Count,Total_Sequences,Percentage\n")
        for main_cat, sub_cats in subcategory_counts.items():
            total = len([idx for idx in range(total_sequences) if idx in category_indices])
            for sub_cat, count in sub_cats.items():
                percentage = count/total*100 if total > 0 else 0
                f.write(f"{main_cat},{sub_cat},{count},{total},{percentage:.2f}\n")
    
    print(f"Subcategory count information saved to {counts_file}")
    
    return subcategory_results, subcategory_counts
