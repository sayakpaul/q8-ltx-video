"""
References:
https://github.com/KONAKONA666/q8_kernels/blob/9cee3f3d4ca5ec8ab463179be32c8001e31f8f33/q8_kernels/utils/convert_weights.py
"""

import torch
from q8_ltx import replace_gelu, replace_linear, replace_rms_norm, MODULES_TO_NOT_CONVERT
import argparse
from diffusers import LTXVideoTransformer3DModel
from q8_kernels.functional.quantizer import quantize
from q8_kernels.functional.fast_hadamard import hadamard_transform


def convert_state_dict(orig_state_dict):
    prefix = "transformer_blocks"
    transformer_block_keys = []
    non_transformer_block_keys = []
    for k in orig_state_dict:
        if prefix in k:
            transformer_block_keys.append(k)
        else:
            non_transformer_block_keys.append(k)
    attn_keys = []
    ffn_keys = []
    scale_shift_keys = []
    for k in transformer_block_keys:
        if "attn" in k:
            attn_keys.append(k)
    for k in transformer_block_keys:
        if "ff" in k:
            ffn_keys.append(k)
    for k in transformer_block_keys:
        if "scale_shift_table" in k:
            scale_shift_keys.append(k)

    assert len(attn_keys + ffn_keys + scale_shift_keys) == len(transformer_block_keys), "error"

    new_state_dict = {}
    for k in attn_keys:
        new_key = k
        if "norm" in k and "weight" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            assert w_quant.dtype == torch.int8, k
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in ffn_keys:
        new_key = k

        if "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            assert w_quant.dtype == torch.int8, k
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in scale_shift_keys:
        new_state_dict[k] = orig_state_dict[k]

    for k in non_transformer_block_keys:
        new_state_dict[k] = orig_state_dict[k]

    return new_state_dict


@torch.no_grad()
def main(args):
    transformer = LTXVideoTransformer3DModel.from_pretrained(args.input_path, subfolder="transformer").to("cuda")
    new_state_dict = convert_state_dict(transformer.state_dict())
    transformer = replace_gelu(transformer)[0]
    transformer = replace_linear(transformer)[0]
    transformer = replace_rms_norm(transformer)[0]

    m, u = transformer.load_state_dict(new_state_dict, strict=True)
    for name, module in transformer.named_modules():
        if any(n in name for n in MODULES_TO_NOT_CONVERT):
            if hasattr(module, "weight"):
                assert module.weight.dtype == torch.float32
            elif hasattr(module, "linear"):
                assert module.linear.weight.dtype == torch.float32
        elif getattr(module, "weight", None) is not None:
            print(f"Non FP32 {name=} {module.weight.dtype=}")
            if "to_" in name:
                assert module.weight.dtype != torch.float32, f"{name=}, {module.weight.dtype=}"

    transformer.save_pretrained(args.output_path)
    print(f"Model saved in {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    main(args)
