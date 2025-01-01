from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from huggingface_hub import hf_hub_download
import argparse
import os
from q8_ltx import check_transformer_replaced_correctly, replace_gelu, replace_linear, replace_rms_norm
import safetensors.torch
from q8_kernels.graph.graph import make_dynamic_graphed_callable
import torch
import gc
from diffusers.utils import export_to_video


# Taken from
# https://github.com/KONAKONA666/LTX-Video/blob/c8462ed2e359cda4dec7f49d98029994e850dc90/inference.py#L115C1-L138C28
def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(char.lower() for char in text if char.isalpha() or char.isspace())
    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)
        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


def load_text_encoding_pipeline():
    return LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to("cuda")


def encode_prompt(pipe, prompt, negative_prompt, max_sequence_length=128):
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt, negative_prompt=negative_prompt, max_sequence_length=max_sequence_length
    )
    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


def load_q8_transformer(args):
    with torch.device("meta"):
        transformer_config = LTXVideoTransformer3DModel.load_config("Lightricks/LTX-Video", subfolder="transformer")
        transformer = LTXVideoTransformer3DModel.from_config(transformer_config)

    transformer = replace_gelu(transformer)[0]
    transformer = replace_linear(transformer)[0]
    transformer = replace_rms_norm(transformer)[0]

    if os.path.isfile(f"{args.q8_transformer_path}/diffusion_pytorch_model.safetensors"):
        state_dict = safetensors.torch.load_file(f"{args.q8_transformer_path}/diffusion_pytorch_model.safetensors")
    else:
        state_dict = safetensors.torch.load_file(
            hf_hub_download(args.q8_transformer_path), filename="diffusion_pytorch_model.safetensors"
        )
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    check_transformer_replaced_correctly(transformer)
    return transformer


@torch.no_grad()
def main(args):
    text_encoding_pipeline = load_text_encoding_pipeline()
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encode_prompt(
        pipe=text_encoding_pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        max_sequence_length=args.max_sequence_length,
    )
    del text_encoding_pipeline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    if args.q8_transformer_path:
        transformer = load_q8_transformer(args)
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=None, text_encoder=None)
        pipe.transformer = transformer

        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        for b in pipe.transformer.transformer_blocks:
            b.to(dtype=torch.float)

        for n, m in pipe.transformer.transformer_blocks.named_parameters():
            if "scale_shift_table" in n:
                m.data = m.data.to(torch.bfloat16)

        pipe.transformer.forward = make_dynamic_graphed_callable(pipe.transformer.forward)
        pipe.vae = pipe.vae.to(torch.bfloat16)

    else:
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", text_encoder=None, torch_dtype=torch.bfloat16)

    pipe = pipe.to("cuda")

    width, height = args.resolution.split("x")[::-1]
    video = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=int(width),
        height=int(height),
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        max_sequence_length=args.max_sequence_length,
        generator=torch.manual_seed(2025),
    ).frames[0]
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB.")

    if args.out_path is None:
        filename_from_prompt = convert_prompt_to_filename(args.prompt, max_len=30)
        base_filename = f"{filename_from_prompt}_{args.num_frames}x{height}x{width}"
        base_filename += "_q8" if args.q8_transformer_path is not None else ""
        args.out_path = base_filename + ".mp4"
    export_to_video(video, args.out_path, fps=24)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8_transformer_path", type=str, default=None)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--resolution", type=str, default="480x704")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
