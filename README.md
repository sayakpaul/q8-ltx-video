# Q8 LTX-Video optimized for Ada

This repository shows how to use the Q8 kernels from [`KONAKONA666/q8_kernels`](https://github.com/KONAKONA666/q8_kernels) with `diffusers` to optimize inference of [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) on ADA GPUs. Go from 16.192 secs to 9.572 while reducing memory from 7GBs to 5GBs without quality loss 🤪

The Q8 transformer checkpoint is available here: [`sayakpaul/q8-ltx-video`](https://hf.co/sayakpaul/q8-ltx-video).

## Getting started

Install the dependencies:

```bash
pip install -U transformers accelerate
git clone https://github.com/huggingface/diffusers && cd diffusers && pip install -e . && cd ..
```

Then install `q8_kernels`, following instructions from [here](https://github.com/KONAKONA666/q8_kernels/?tab=readme-ov-file#installation).

To run inference with the Q8 kernels, we need some minor changes in `diffusers`. Apply [this patch](https://github.com/sayakpaul/q8-ltx-video/blob/368f549ca5136daf89049c9efe32748e73aca317/updates.patch) to take those into account:

```bash
git apply updates.patch
```

Now we can run inference:

```bash
python inference.py \
    --prompt="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
    --negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted" \
    --q8_transformer_path="sayakpaul/q8-ltx-video"
```

## Why does the repo exist and some more details?

There already exists [`KONAKONA666/LTX-Video`](https://github.com/KONAKONA666/LTX-Video). Then why this repo?

That repo uses custom implementations of the LTX-Video pipeline components and can be hard to directly use in `diffusers`. This repo repurposes the kernels from the `q8_kernels` on the components directly from `diffusers`.

<details>
<summary>More details</summary>

We do this by first converting the state dict of the original [LTX-Video transformer](https://huggingface.co/Lightricks/LTX-Video/tree/main/transformer). This includes FP8 quantization. This process also requires replacing:

* linear layers of the model
* RMSNorms of the model
* GELUs of the model

before the converted state dict is loaded into the model. Some layer params are kept in FP32 and some layers are not even quantized. Replacement utilities are in [`q8_ltx.py`](./q8_ltx.py).

The model can then be serialized. The conversion and serialization are coded in [`conversion_utils.py`](./conversion_utils.py).

During loading the model and using it for inference, we:

* initialize the transformer model under a "meta" device
* follow the same layer replacement scheme as detailed above
* populate the converted state dict

Refer [here](https://github.com/sayakpaul/q8-ltx-video/blob/368f549ca5136daf89049c9efe32748e73aca317/inference.py#L48) more details. Additionally, we leverage [flash-attention implementation](https://github.com/sayakpaul/q8-ltx-video/blob/368f549ca5136daf89049c9efe32748e73aca317/q8_attention_processors.py#L44) from `q8_kernels` which provides further speedup.

</details>

## Performance

Below numbers were obtained for `max_sequence_length=128`, `num_inference_steps=50`, `num_frames=81`, `resolution=480x704`.


|  | **Time (Secs)** | **Memory (MB)** |
|:-----------:|:-----------:|:-----------:|
| Non Q8  | 16.192 | 7172.86  |
| Q8  | 9.572 secs  | 5413.51  |

Benchmarking script is available in [`benchmark.py`](./benchmark.py).

<details>
<summary>Env</summary>

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0 Off |                  Off |
|  0%   46C    P8             18W /  450W |       2MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

`diffusers-cli env`:

```bash
- 🤗 Diffusers version: 0.33.0.dev0
- Platform: Linux-6.8.0-49-generic-x86_64-with-glibc2.39
- Running on Google Colab?: No
- Python version: 3.10.12
- PyTorch version (GPU?): 2.5.1+cu124 (True)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Huggingface_hub version: 0.27.0
- Transformers version: 4.47.1
- Accelerate version: 1.2.1
- PEFT version: 0.13.2
- Bitsandbytes version: 0.44.1
- Safetensors version: 0.4.4
- xFormers version: 0.0.29.post1
- Accelerator: NVIDIA GeForce RTX 4090, 24564 MiB
NVIDIA GeForce RTX 4090, 24564 MiB
- Using GPU in script?: <fill in>
- Using distributed or parallel set-up in script?: <fill in>
```

</details>

> [!NOTE]
> The RoPE implementation [isn't usable as of 1st Jan 2025](https://github.com/KONAKONA666/q8_kernels/blob/9cee3f3d4ca5ec8ab463179be32c8001e31f8f33/q8_kernels/functional/rope.py#L26). So, we resort to using [the one](https://github.com/huggingface/diffusers/blob/91008aabc4b8dbd96a356ab6f457f3bd84b10e8b/src/diffusers/models/transformers/transformer_ltx.py#L464) from `diffusers`.


## Comparison

Check out [this page](https://wandb.ai/sayakpaul/q8-ltx-video/runs/89h6ac5) on Weights and Biases that provides some comparative results. Generated videos are also available [here](./videos/).

## Acknowledgement

KONAKONA666's works on [`KONAKONA666/q8_kernels`](https://github.com/KONAKONA666/q8_kernels) and [KONAKONA666/LTX-Video](https://github.com/KONAKONA666/LTX-Video).

