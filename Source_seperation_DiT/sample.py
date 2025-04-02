# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from data import MultiSourceDataset
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from torch.utils.data import DataLoader
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    testset = MultiSourceDataset(
    sr=22050,
    channels=1,
    min_duration=12,
    max_duration=640,
    aug_shift=True,
    sample_length=262144,
    audio_files_dir="/home/jingyi49/multi-source-diffusion-models/data/slakh2100/test",
    stems=['bass', 'drums', 'guitar', 'piano'],)
    
    dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    

    # Create sampling noise:
    #n = len(testset)
    n = 1
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    #y = torch.tensor(class_labels, device=device)
    for data in dataloader:
        x1 = data[:, 0, :].squeeze().reshape(1, 512, 512)
        x2 = data[:, 1, :].squeeze().reshape(1, 512, 512)
        x3 = data[:, 2, :].squeeze().reshape(1, 512, 512)
        x4 = data[:, 3, :].squeeze().reshape(1, 512, 512)
        #将4个乐器的图像加在一起
        y = torch.sum(torch.stack([x1, x2, x3]), dim=0)
        y = y.to(device)
        break

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    #y_null的shape和y的shape一样，但是y_null的值全为1000
    y_null = torch.full_like(y, 1000, device=y.device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
