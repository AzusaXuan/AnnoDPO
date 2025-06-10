import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
from urllib.parse import urlparse
from models.protein_bert_seq import ANNO_BERT, ANNO_Wrapper, SEQ_BERT, SEQ_Wrapper


def create_proteinBERT(bert='seq_bert', args=None):
    if bert.startswith('seq_bert'):
        protein_bert = SEQ_BERT(
            num_tokens=args.num_tokens,
            dim=args.dim,
            dim_global=args.dim_global,
            depth=args.depth,
            narrow_conv_kernel=args.narrow_conv_kernel,
            wide_conv_kernel=args.wide_conv_kernel,
            wide_conv_dilation=args.wide_conv_dilation,
            attn_heads=args.attn_heads,
            attn_dim_head=args.attn_dim_head,
            local_self_attn=args.local_self_attn,
            glu_conv=args.glu_conv,
        )
        seq_encoder = SEQ_Wrapper(
            protein_bert,
            model_type=bert,
            args=args
        )

    elif bert.startswith('anno_bert'):

        protein_bert = ANNO_BERT(
            # num_tokens=args.num_tokens,
            num_annotation=args.num_annotation,
            dim=args.dim,
            dim_global=args.dim_global,
            depth=args.depth,
            attn_heads=args.attn_heads,
            attn_dim_head=args.attn_dim_head,
            num_global_tokens=args.num_global_tokens,
        )
        seq_encoder = ANNO_Wrapper(
            protein_bert,
            model_type=bert
        )

    else:
        raise RuntimeError('bert parameter must be seq_bert or anno_bert')

    return seq_encoder


