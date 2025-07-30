# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch

# === guard flash_attn import ===
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaFlashAttention2
)


# Modified from transformers.models.llama.modeling_llama.LlamaFlashAttention2
class LlamaFlashAttention2ForPackedTraining(LlamaFlashAttention2):

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        # if flash_attn isn't available, fall back to the parent class behavior
        if flash_attn_varlen_func is None:
            return super()._flash_attention_forward(
                query_states, key_states, value_states,
                attention_mask, query_length,
                dropout=dropout,
                softmax_scale=softmax_scale,
                use_sliding_windows=use_sliding_windows
            )

        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)

        with torch.no_grad():
            max_seqlen = max([
                cu_seqlens[idx+1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]).item()

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # keep the legacy check for RoCm <2.1
            causal = self.is_causal and query_length != 1

        if use_sliding_windows and self.layer_idx < self.config.max_window_layers:
            # only use SWA on early layers
            use_sliding_windows = True
        else:
            use_sliding_windows = False

        # call FlashAttention
        if not use_sliding_windows:
            attn_output = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            attn_output = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

        return attn_output

def replace_llama_attention_class():
    if flash_attn_varlen_func is None:
        print("[internvl] flash_attn not found; skipping packed-training patch.")
        return

    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaFlashAttention2ForPackedTraining
    print('Replaced LLAMA_ATTENTION_CLASSES to support packed training!!')
