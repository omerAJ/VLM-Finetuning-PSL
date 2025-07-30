import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from internvl.model.internlm2.modeling_internlm2 import (
    INTERNLM2_ATTENTION_CLASSES, InternLM2FlashAttention2,
    apply_rotary_pos_emb,
)

class InternLM2FlashAttention2ForPackedTraining(InternLM2FlashAttention2):
    def __init__(self, config):
        super().__init__(config)
        # we'll reuse a single MultiheadAttention for fallback
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        self._fallback_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=config.attention_dropout, batch_first=True)

    def _flash_attention_forward(
        self, query_states, key_states, value_states,
        attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # unpack the packed-flash interface args
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        q = query_states.squeeze(0)   # (total_tokens, embed_dim)
        k = key_states.squeeze(0)
        v = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)  # (batch+1,)

        # if we have FlashAttention, use it
        if flash_attn_varlen_func is not None:
            max_seqlen = max(
                cu_seqlens[i+1] - cu_seqlens[i]
                for i in range(cu_seqlens.size(0)-1)
            ).item()
            causal = self.is_causal and (query_length != 1)
            out = flash_attn_varlen_func(
                q=q, k=k, v=v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=dropout, softmax_scale=softmax_scale, causal=causal,
            )
            return out

        # --- FALLBACK PATH: pad to dense and run MultiheadAttention ---
        # figure out batch size and max length
        batch_size = cu_seqlens.size(0) - 1
        seq_lens = [(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(batch_size)]
        max_len = max(seq_lens)

        # split and pad each segment
        segments = []
        idx = 0
        for L in seq_lens:
            seg = q[idx:idx+L]
            idx += L
            # pad to max_len
            if L < max_len:
                pad = torch.zeros((max_len - L, q.size(-1)), device=q.device, dtype=q.dtype)
                seg = torch.cat([seg, pad], dim=0)
            segments.append(seg)
        
        # stack into (batch, max_len, embed_dim)
        q_pad = torch.stack(segments, dim=0)
        k_pad = q_pad  # packed queries == keys
        v_pad = q_pad  # packed values == values

        # positional embeddings (if needed)
        # q_pad, k_pad = apply_rotary_pos_emb(q_pad, k_pad, ...)

        # run dense MHA
        # Note: MultiheadAttention returns (output, attn_weights)
        out_pad, _ = self._fallback_mha(q_pad, k_pad, v_pad, need_weights=False, attn_mask=None)

        # now unpad back to original packed shape
        outputs = []
        for i, L in enumerate(seq_lens):
            outputs.append(out_pad[i, :L, :])
        return torch.cat(outputs, dim=0).unsqueeze(0)

def replace_internlm2_attention_class():
    INTERNLM2_ATTENTION_CLASSES['flash_attention_2'] = InternLM2FlashAttention2ForPackedTraining
    print('Replaced INTERNLM2_ATTENTION_CLASSES to support packed training with fallback!')
