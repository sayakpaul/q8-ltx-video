"""
References:
https://github.com/KONAKONA666/q8_kernels/blob/9cee3f3d4ca5ec8ab463179be32c8001e31f8f33/q8_kernels/utils/convert_weights.py
"""

import torch
import torch.nn as nn
from q8_kernels.modules.rms_norm import RMSNorm as QRMSNorm
from diffusers.models.normalization import RMSNorm
from q8_kernels.modules.activations import GELU as QGELU
from diffusers.models.activations import GELU
from q8_kernels.modules.linear import Q8Linear
from q8_attention_processors import LTXVideoQ8AttentionProcessor

MODULES_TO_NOT_CONVERT = ["proj_in", "time_embed", "caption_projection", "proj_out"]


def replace_linear(model, current_key_name=None, replaced=False):
    for name, child in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(child, nn.Linear) and name not in MODULES_TO_NOT_CONVERT:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in MODULES_TO_NOT_CONVERT
            ):
                new_linear = Q8Linear(
                    child.in_features, child.out_features, bias=child.bias is not None, device=child.weight.device
                )
                setattr(model, name, new_linear)
                replaced = True
        else:
            replace_linear(model=child, current_key_name=current_key_name, replaced=replaced)

        current_key_name.pop(-1)

    return model, replaced


def get_parent_module_and_attr(root, dotted_name: str):
    """
    Splits 'a.b.c' into:
    - parent module = root.a.b
    - attr_name = 'c'
    """
    parts = dotted_name.split(".")
    *parent_parts, attr_name = parts
    parent_module = root
    for p in parent_parts:
        parent_module = getattr(parent_module, p)
    return parent_module, attr_name


def replace_rms_norm(model):
    modules_to_replace = []
    for dotted_name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            modules_to_replace.append((dotted_name, module))

    replaced = False
    for dotted_name, module in modules_to_replace:
        parent, attr_name = get_parent_module_and_attr(model, dotted_name)
        new_norm = QRMSNorm(
            dim=module.dim,
            elementwise_affine=module.elementwise_affine,
        )
        setattr(parent, attr_name, new_norm)
        replaced = True

    return model, replaced


def replace_gelu(model, replaced=False):
    for name, child in model.named_children():
        if isinstance(child, GELU):
            new_gelu = QGELU(
                dim_in=child.proj.in_features,
                dim_out=child.proj.out_features,
                approximate=child.approximate,
                bias=child.proj.bias is not None,
            )
            setattr(model, name, new_gelu)
            replaced = True
        else:
            replace_gelu(model=child, replaced=replaced)

    return model, replaced


def set_attn_processors(model, processor):
    def fn_recursive_attn_processor(name, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            module.set_processor(processor)
        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor)


def attn_processors(model) -> dict:
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: dict):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = module.get_processor()

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def check_transformer_replaced_correctly(model):
    for block in model.transformer_blocks:
        assert isinstance(block.attn1.to_q, Q8Linear), f"{type(block.attn1.to_q)=} not linear."
        assert isinstance(block.attn2.to_q, Q8Linear), f"{type(block.attn2.to_q)=} not linear."
        assert block.attn1.to_q.weight.dtype == torch.int8, f"{block.attn1.to_q.weight.dtype=}."
        assert block.attn2.to_q.weight.dtype == torch.int8, f"{name=} {block.attn2.to_q.weight.dtype=}."

    for name, module in model.named_modules():
        if "norm" in name and "norm_out" not in name:
            assert isinstance(module, QRMSNorm), f"{name=}, {type(module)=}"

    for block in model.transformer_blocks:
        assert isinstance(block.ff.net[0], QGELU), f"{type(block.ff.net[0])=}"
        if getattr(block.ff.net[0], "proj", None) is not None:
            assert block.ff.net[0].proj.weight.dtype == torch.int8, f"{block.ff.net[0].proj.weight.dtype=}."

    set_attn_processors(model, LTXVideoQ8AttentionProcessor())
    all_attn_processors = attn_processors(model)
    for k, v in all_attn_processors.items():
        assert isinstance(v, LTXVideoQ8AttentionProcessor), f"{name} is not of type LTXVideoQ8AttentionProcessor."
