"""
Reference:
https://github.com/KONAKONA666/q8_kernels/blob/9cee3f3d4ca5ec8ab463179be32c8001e31f8f33/q8_kernels/modules/attention.py
"""

import torch
import q8_kernels.functional as Q8F
from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
from diffusers.models.attention import Attention

NON_MM_PRECISION_TYPE = torch.bfloat16
MM_PRECISION_TYPE = torch.bfloat16

class LTXVideoQ8AttentionProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        image_rotary_emb = None,
    ) -> torch.Tensor:
        if attention_mask is not None and  attention_mask.ndim > 1:
            attention_mask = attention_mask.argmin(-1).squeeze().int()

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query, NON_MM_PRECISION_TYPE)
        key = attn.norm_k(key, NON_MM_PRECISION_TYPE)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = Q8F.flash_attention.flash_attn_func(
            query,
            key,
            value,
            batch_mask=attention_mask,
            apply_qk_hadamard=True
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states.to(NON_MM_PRECISION_TYPE)