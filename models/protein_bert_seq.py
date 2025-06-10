import math
import sys

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
import wandb


# helpers

def exists(val):
    return val is not None


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class GlobalLinearSelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head,
            heads
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask=None):
        B_, N, _ = feats.shape
        h = self.heads
        qkv = self.to_qkv(feats).reshape(B_, N, 3, h, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if exists(mask):
            mask = mask.unsqueeze(1).unsqueeze(-1)
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = k.transpose(-2, -1) @ v
        out = (q @ context.transpose(-2, -1)).transpose(1, 2).flatten(-2)

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_keys,
            dim_out,
            heads,
            dim_head=64,
            qk_activation=nn.Tanh()
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))
        # print(f'dim: {dim}, dim_keys: {dim_keys}, dim_out: {dim_out}, heads: {heads}, dim_head: {dim_head}')

    def forward(self, x, context, mask=None, context_mask=None):
        # print(f'x.shape: {x.shape}, context.shape: {context.shape}')
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        B, N, D = q.shape
        B_, N_, D_ = k.shape
        d = D // h
        d_ = D_ // h

        q = q.reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = k.reshape(B_, N_, h, d_).permute(0, 2, 1, 3)
        v = v.reshape(B_, N_, h, d_).permute(0, 2, 1, 3)

        null_k = self.null_key.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(b, h, 1, -1)
        null_v = self.null_value.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(b, h, 1, -1)

        k = torch.cat((null_k, k), dim=-2)
        v = torch.cat((null_v, v), dim=-2)

        q = self.qk_activation(q)
        k = self.qk_activation(k)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype=torch.bool, device=device)

            if exists(context_mask):
                context_mask = F.pad(context_mask, (1, 0), value=True)
            else:
                context_mask = torch.ones(b, j, dtype=torch.bool, device=device)

            mask = mask.unsqueeze(1).unsqueeze(-1) * context_mask.unsqueeze(1).unsqueeze(1)
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).flatten(-2)

        return self.to_out(out)


class SEQ_Layer_v2(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_global,
            narrow_conv_kernel=9,
            wide_conv_kernel=9,
            wide_conv_dilation=5,
            attn_heads=8,
            attn_dim_head=64,
            # attn_qk_activation=nn.Tanh(),
            # local_to_global_attn=False,
            local_self_attn=False,
            glu_conv=False
    ):
        super().__init__()

        self.seq_self_attn = GlobalLinearSelfAttention(dim=dim, dim_head=attn_dim_head,
                                                       heads=attn_heads) if local_self_attn else None

        conv_mult = 2 if glu_conv else 1

        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding=narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim=1)
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation=wide_conv_dilation, padding=wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim=1)
        )

        self.extract_global_info = nn.Sequential(
            # Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim_global, dim),
            nn.GELU(),
            Rearrange('b d -> b () d')
        )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )),
            nn.LayerNorm(dim)
        )

        # self.local_dense = nn.Sequential(
        #     nn.Linear(dim_global, dim_global),
        #     nn.GELU()
        # )

    def forward(self, tokens, anno_embed=None, mask=None):

        self_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0
        conv_input = tokens.transpose(-2, -1)

        if exists(mask):
            conv_input_mask = mask.unsqueeze(1)
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input).transpose(-2, -1)
        wide_out = self.wide_conv(conv_input).transpose(-2, -1)
        if anno_embed is not None:
            # print("anno embed: ", anno_embed)
            # sys.exit()
            global_info = self.extract_global_info(anno_embed)

            tokens = tokens + narrow_out + wide_out + global_info + self_linear_attn
            # tokens = tokens + narrow_out + wide_out + global_info
        else:
            tokens = tokens + narrow_out + wide_out + self_linear_attn
            # tokens = tokens + narrow_out + wide_out

        ## Norm and FF after getting tokens
        tokens = self.local_norm(tokens)
        tokens = self.local_feedforward(tokens)

        return tokens


class ANNO_Layer_v2(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_global,
            # narrow_conv_kernel=9,
            # wide_conv_kernel=9,
            # wide_conv_dilation=5,
            attn_heads=8,
            attn_dim_head=64,
            attn_qk_activation=nn.Tanh(),
            # local_to_global_attn=False,
            # local_self_attn=False,
            # glu_conv=False
    ):
        super().__init__()
        # self.global_attend_local = GlobalLinearSelfAttention(dim=dim_global, dim_head=attn_dim_head,
        #                                                      heads=attn_heads)

        self.global_cross_atten = CrossAttention(dim=dim_global, dim_out=dim_global, dim_keys=dim, heads=attn_heads,
                                                 dim_head=attn_dim_head, qk_activation=attn_qk_activation)

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU()
        )

        self.global_norm = nn.LayerNorm(dim_global)

        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim_global, dim_global),
                nn.GELU()
            )),
            nn.LayerNorm(dim_global),
        )

    def forward(self, annotation, seq_embed=None, mask=None):
        # self_attn_annotation = self.global_attend_local(annotation)
        dense = self.global_dense(annotation)
        if seq_embed is not None:
            global_atten = self.global_cross_atten(annotation, seq_embed)
            annotation = annotation + global_atten + dense  # + self_attn_annotation
        else:
            annotation = annotation + dense  # + self_attn_annotation
        annotation = self.global_norm(annotation)

        annotation = self.global_feedforward(annotation)
        return annotation


# main model

class SEQ_BERT(nn.Module):
    def __init__(
            self,
            *,
            num_tokens=26,
            dim=512,
            dim_global=256,
            depth=6,
            narrow_conv_kernel=9,
            wide_conv_kernel=9,
            wide_conv_dilation=5,
            attn_heads=4,
            attn_dim_head=64,
            local_self_attn=False,
            glu_conv=False
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([SEQ_Layer_v2(dim=dim, dim_global=dim_global, narrow_conv_kernel=narrow_conv_kernel,
                                                  wide_conv_dilation=wide_conv_dilation,
                                                  wide_conv_kernel=wide_conv_kernel,
                                                  local_self_attn=local_self_attn,
                                                  attn_heads=attn_heads,
                                                  attn_dim_head=attn_dim_head,
                                                  glu_conv=glu_conv) for layer in range(depth)])

    def forward(self, seq, annotation_embed=None, mask=None):
        '''
        seq: [bs, seq len]
        '''
        tokens = self.token_emb(seq)  # seq: [bs, seq len, dim]

        for layer in self.layers:
            tokens = layer(tokens, annotation_embed, mask=mask)

        return tokens


class SEQ_Wrapper(nn.Module):
    def __init__(
            self,
            model,
            model_type="seq_bert_itc",
            args=None
    ):
        super().__init__()

        self.model = model
        self.model_type = model_type
        # self.random_replace_token_prob = args.random_replace_token_prob
        # self.exclude_token_ids = args.exclude_token_ids
        # self.num_tokens = args.num_tokens

    def forward(self, seq, mask=None, anno_embed=None):

        if self.model_type == "seq_bert_lm":  # seq decoder
            seq_logits = self.model(seq, annotation_embed=anno_embed, mask=mask)

        elif self.model_type == "seq_bert_itc":  # seq encoder
            seq_logits = self.model(seq, mask=mask)

        return seq_logits


class ANNO_BERT(nn.Module):
    def __init__(
            self,
            *,
            num_annotation=7533,
            dim=128,
            dim_global=512,
            depth=6,
            attn_heads=4,
            attn_dim_head=64,
            attn_qk_activation=nn.Tanh(),
            num_global_tokens=1,
    ):
        super().__init__()

        self.num_global_tokens = num_global_tokens
        self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)

        self.layers = nn.ModuleList([ANNO_Layer_v2(dim=dim, dim_global=dim_global,
                                                   attn_qk_activation=attn_qk_activation,
                                                   attn_heads=attn_heads,
                                                   attn_dim_head=attn_dim_head,
                                                   ) for layer in range(depth)])
        self.to_annotation_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
        )

    def forward(self, annotation, seq_embed=None, mask=None):
        annotation = self.to_global_emb(annotation)
        annotation = annotation.reshape(annotation.size(0), self.num_global_tokens, -1)
        for layer in self.layers:
            annotation = layer(annotation, seq_embed=seq_embed, mask=mask)

        annotation = self.to_annotation_logits(annotation)
        return annotation


class ANNO_Wrapper(nn.Module):
    def __init__(
            self,
            model,
            model_type='anno_bert_itc'
    ):
        super().__init__()

        self.model = model
        self.model_type = model_type

    def forward(self, annotation, mask=None, seq_embed=None):
        if self.model_type == 'anno_bert_itc':
            annotation_logits = self.model(annotation, seq_embed=seq_embed, mask=mask)
        elif self.model_type == 'anno_bert_lm':
            # masked_annotation = torch.zeros(annotation.shape, device=annotation.device)
            annotation_logits = self.model(annotation, seq_embed=seq_embed, mask=mask)

        return annotation_logits
