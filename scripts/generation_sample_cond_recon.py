"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
from scipy import ndimage

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from guided_diffusion.toothloader import ToothVolumes
from torch.utils.data import DataLoader


def compute_gradient_mag(volume: th.Tensor) -> th.Tensor:
    dwt = DWT_3D("haar")
    with th.no_grad():
        LLL, *_ = dwt(volume)
    LLL_np = LLL.squeeze().cpu().numpy()

    grad_x = ndimage.sobel(LLL_np, axis=0)
    grad_y = ndimage.sobel(LLL_np, axis=1)
    grad_z = ndimage.sobel(LLL_np, axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    grad_mag_norm = 2 * (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8) - 1
    grad_heatmap = th.tensor(grad_mag_norm, dtype=th.float32).unsqueeze(0).unsqueeze(0)
    return grad_heatmap


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(f"Loading model from: {args.model_path}")
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    model.eval()

    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = ToothVolumes(
        directory=args.data_dir,
        metadata_path=args.meta_data,
        test_flag=True,
        mode='eval',
        normalize=(lambda x: 2 * x - 1),
        img_size=args.image_size,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    base_sample = None

    for sample in loader:
        missing = (sample['tooth_presence'] == 0).sum().item()
        if missing > 5:
            base_sample = sample
            logger.log(f"Selected sample: {sample['name'][0]} with {missing} missing teeth.")
            break

    if base_sample is None:
        raise RuntimeError("No suitable sample with ≥5 missing teeth found.")

    base_sample = {k: v.to(dist_util.dev()) if k != 'name' else v for k, v in base_sample.items()}

    raw_name = base_sample['name'][0]
    if isinstance(raw_name, tuple):
        raw_name = raw_name[0]
    base_name = os.path.splitext(os.path.splitext(raw_name)[0])[0]

    base_image = base_sample['image']
    base_tp = base_sample['tooth_presence']
    base_crown_fill = base_sample['crown_fill']
    base_root_crown = base_sample['root_crown']
    base_bridge = base_sample['bridge']
    base_implant = base_sample['implant']

    scenarios = ["implant", "crown", "root_crown", "bridge"]

    for scenario in scenarios:
        logger.log(f"Running scenario: {scenario}")

        tooth_presence = base_tp.clone()
        crown_fill = th.zeros_like(tooth_presence)
        root_crown = th.zeros_like(tooth_presence)
        bridge = th.zeros_like(tooth_presence)
        implant = th.zeros_like(tooth_presence)

        present_indices = (tooth_presence[0] > 0).nonzero(as_tuple=True)[0].tolist()
        if not present_indices:
            raise RuntimeError("No teeth present to modify.")

        logger.log(f"Setting all present teeth as {scenario}")
        if scenario == "implant":
            implant[0, present_indices] = 1.0
        elif scenario == "crown":
            crown_fill[0, present_indices] = 1.0
        elif scenario == "root_crown":
            root_crown[0, present_indices] = 1.0
        elif scenario == "bridge":
            bridge[0, present_indices] = 1.0

        LLL_gradient_mag = compute_gradient_mag(base_image).to(dist_util.dev())

        LLL_img, LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img = dwt(base_image)
        cond_dwt = th.cat([LLL_gradient_mag, LLL_img / 3., LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img], dim=1)

        model_kwargs = {
            "tooth_presence": tooth_presence,
            "crown_fill": crown_fill,
            "root_crown": root_crown,
            "bridge": bridge,
            "implant": implant,
        }

        img = th.randn(args.batch_size,
                       8,
                       args.image_size//2,
                       args.image_size//2,
                       args.image_size//2).to(dist_util.dev())

        sample_out = diffusion.p_sample_loop(
            model=model,
            shape=img.shape,
            noise=img,
            cond=cond_dwt,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        B, _, D, H, W = sample_out.size()
        sample_out = idwt(
            sample_out[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
            sample_out[:, 1, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 2, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 3, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 4, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 5, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 6, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 7, :, :, :].view(B, 1, D, H, W))

        sample_out = (sample_out + 1) / 2.

        if len(sample_out.shape) == 5:
            sample_out = sample_out.squeeze(1)
            
        output_dir = os.path.join(args.output_dir, os.path.basename(args.model_path).replace('.pt', ''))
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_name = os.path.join(output_dir, f"sample_{base_name}_{scenario}.nii.gz")
        img_nii = nib.Nifti1Image(sample_out[0].detach().cpu().numpy(), np.eye(4))
        nib.save(img_nii, filename=output_name)
        logger.log(f"Saved to {output_name}")


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        meta_data="",
        model_path="",
        devices=[0],
        output_dir='./results',
        clip_denoised=True,
        batch_size=1,
        image_size=256,
    )

    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
