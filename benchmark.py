from diffusers import LTXPipeline
import torch.utils.benchmark as benchmark_pt
import torch
import json
import argparse


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark_pt.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


def run_inference(pipe, prompt_embeds, **kwargs):
    _ = pipe(**prompt_embeds, generator=torch.manual_seed(0), output_type="latent", **kwargs)


def run_benchmark(pipe, args, prompt_embeds):
    for _ in range(5):
        run_inference(pipe, prompt_embeds)

    width, height = args.resolution.split("x")[::-1]
    time = benchmark_fn(
        run_inference, pipe, prompt_embeds, num_frames=args.num_frames, width=int(width), height=int(height)
    )

    info = dict(num_frames=args.num_frames, width=int(width), height=int(height), time=time)
    path = f"{args.num_frames}x{height}x{width}"
    path += "_q8" if args.q8_transformer_path else ""
    path += ".json"
    with open(path, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8_transformer_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--resolution", type=str, default="480x704")
    args = parser.parse_args()

    if args.q8_transformer_path is None:
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", text_encoder=None, vae=None, torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        from inference import load_q8_transformer
        from q8_kernels.graph.graph import make_dynamic_graphed_callable

        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", text_encoder=None, transformer=None, vae=None)
        transformer = load_q8_transformer(args)
        pipe.transformer = transformer

        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        for b in pipe.transformer.transformer_blocks:
            b.to(dtype=torch.float)

        for n, m in pipe.transformer.transformer_blocks.named_parameters():
            if "scale_shift_table" in n:
                m.data = m.data.to(torch.bfloat16)

        pipe.transformer.forward = make_dynamic_graphed_callable(pipe.transformer.forward)
        pipe = pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    prompt_embeds = torch.load("prompt_embeds.pt", map_location="cuda", weights_only=True)
    run_benchmark(pipe, args, prompt_embeds)
