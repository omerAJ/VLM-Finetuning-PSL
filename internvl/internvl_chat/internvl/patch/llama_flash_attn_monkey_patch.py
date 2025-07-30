"""
This file is copied from: https://github.com/lm-sys/FastChat
"""

import warnings
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# === Try to import flash_attn; if missing, we'll skip flash paths ===
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func,
        flash_attn_varlen_qkvpacked_func as flash_attn_func_v2,
    )
    from flash_attn.bert_padding import pad_input, unpad_input
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False
    pad_input = unpad_input = flash_attn_unpadded_qkvpacked_func = flash_attn_func_v2 = None

def forward_flash(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """FlashAttention-based forward; assumes flash_attn is present."""
    from einops import rearrange

    bsz, q_len, _ = hidden_states.size()
    nh, hd = self.num_heads, self.head_dim

    # project to q/k/v
    q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)

    # rotary
    cos, sin = self.rotary_emb(v, seq_len=q_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    # pack into [bsz, q_len, 3, nh, hd]
    qkv = rearrange(torch.stack([q, k, v], dim=2), 'b h three s d -> b s three h d')
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        x = rearrange(qkv, 'b s three h d -> (b s) three h d')
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len,
                                 step=q_len, dtype=torch.int32, device=x.device)
        out = flash_attn_unpadded_qkvpacked_func(x, cu_q_lens, q_len,
                                                 0.0, softmax_scale=None,
                                                 causal=True)
        out = rearrange(out, '(b s) h d -> b s h d', b=bsz)
    else:
        nheads = nh
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unp, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unp = rearrange(x_unp, 'nnz (three h d) -> nnz three h d',
                          three=3, h=nheads)
        out_unp = flash_attn_unpadded_qkvpacked_func(x_unp, cu_q_lens,
                                                     max_s, 0.0,
                                                     softmax_scale=None,
                                                     causal=True)
        out = rearrange(
            pad_input(rearrange(out_unp, 'nnz h d -> nnz (h d)'),
                      indices, bsz, q_len),
            'b s (h d) -> b s h d', h=nheads
        )

    out = self.o_proj(rearrange(out, 'b s h d -> b s (h d)'))
    return out, None, None


def forward_pytorch(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Pure-PyTorch fallback using scaled_dot_product_attention when available."""
    bsz, q_len, _ = hidden_states.size()
    nh, hd = self.num_heads, self.head_dim

    # project to q/k/v
    q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)

    # rotary
    cos, sin = self.rotary_emb(v, seq_len=q_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    if past_key_value is not None:
        # not supported in this fallback
        raise NotImplementedError("past_key_value not supported in PyTorch fallback")

    if hasattr(F, "scaled_dot_product_attention"):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=True
        )
        out = out.transpose(1, 2).reshape(bsz, q_len, nh * hd)
    else:
        # manual matmul + masking + softmax
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, :, :],
                                        float('-inf'))
        weights = scores.softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(bsz, q_len, nh * hd)

    return self.o_proj(out), None, None


def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    return attention_mask


def replace_llama_attn_with_flash_attn():
    """
    Installs:
      - LlamaModel._prepare_decoder_attention_mask → our passthrough
      - LlamaAttention.forward → flash or pytorch, depending on availability
    """
    # always override the mask prep
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask

    if _HAS_FLASH:
        print("[internvl] flash_attn available → using FlashAttention forward")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flash
    else:
        print("[internvl] flash_attn not found → using PyTorch fallback forward")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_pytorch
