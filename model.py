from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba as MambaBlock

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    input_dim: int
    output_dim: int
    d_state: int = 64
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaPretrainBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.input_dim, args.d_model)
        self.layers = nn.ModuleList([MambaBlock(
            d_model=args.d_model
        ) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + self.norm_f(layer(x))
        return x

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.input_dim, args.d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model=args.d_model) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.output_dim, bias=False)


    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + self.norm_f(layer(x))
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits


    def load_params(self,layer_ids,load_path,frozentype):
        state_dict = torch.load(load_path)
        for i in layer_ids:
            self.layers[i].load_params(state_dict,frozentype)
    
    
class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
