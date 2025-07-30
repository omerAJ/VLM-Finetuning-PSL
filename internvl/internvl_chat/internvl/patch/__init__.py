# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Determine if flash_attn is installed
try:
    import flash_attn
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

# Always-available patches
from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit
from .pad_data_collator import (
    concat_pad_data_collator,
    dpo_concat_pad_data_collator,
    pad_data_collator
)
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

# Conditionally import or stub flash-attn dependent patches
if _HAS_FLASH:
    from .internlm2_packed_training_patch import replace_internlm2_attention_class
    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
    from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    from .llama_packed_training_patch import replace_llama_attention_class
    from .phi3_packed_training_patch import replace_phi3_attention_class
    from .qwen2_packed_training_patch import replace_qwen2_attention_class
else:
    # flash_attn not available: define no-op stubs
    def replace_internlm2_attention_class():
        print('[internvl] flash_attn not installed; skipping internlm2 patch.')

    def replace_llama2_attn_with_flash_attn():
        print('[internvl] flash_attn not installed; skipping llama2 flash patch.')

    def replace_llama_attn_with_flash_attn():
        print('[internvl] flash_attn not installed; skipping llama flash patch.')

    def replace_llama_attention_class():
        print('[internvl] flash_attn not installed; skipping llama packed-training patch.')

    def replace_phi3_attention_class():
        print('[internvl] flash_attn not installed; skipping phi3 packed-training patch.')

    def replace_qwen2_attention_class():
        print('[internvl] flash_attn not installed; skipping qwen2 packed-training patch.')

# Public API
__all__ = [
    'apply_liger_kernel_to_internvit',
    'replace_train_sampler',
    'replace_train_dataloader',
    'concat_pad_data_collator',
    'dpo_concat_pad_data_collator',
    'pad_data_collator',
    'replace_internlm2_attention_class',
    'replace_qwen2_attention_class',
    'replace_phi3_attention_class',
    'replace_llama_attention_class',
    'replace_llama2_attn_with_flash_attn',
    'replace_llama_attn_with_flash_attn',
]
