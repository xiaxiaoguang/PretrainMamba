"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba as MambaB

@dataclass
class ModelArgs:
    d_model: int
    e_fact: int = 1
    e_layers: int = 0
    revin: bool = False
    threshold : float = 0.6
    d_state: int = 16
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 2
    n_layer : int = 0
    input_dim : int = 0
    output_dim : int= 0
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.e_fact * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaPre(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.input_dim, args.d_model)
        self.layers = nn.ModuleList([MambaB(
            d_model=args.d_model, 
            d_state=args.d_state, 
            d_conv=args.d_conv, 
            expand=args.e_fact,
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
        self.layers = nn.ModuleList([MambaB(d_model=args.d_model,
                                            d_state=args.d_state, 
                                            d_conv=args.d_conv, 
                                            expand=args.e_fact,) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.output_dim, bias=False)


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l, input_dim)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, output_dim)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
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
    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)

        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            d_state=config_data['d_state'], 
            d_conv=config_data['d_conv'], 
            e_fact=config_data['e_fact'],
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model

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
        
