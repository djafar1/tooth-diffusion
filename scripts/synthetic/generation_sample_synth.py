"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import sys
import torch as th

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # Fix DDP prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices


    model.eval()
    idwt = IDWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)

    model_name = os.path.basename(args.model_path).replace('.pt', '')
    output_dir = os.path.join(args.output_dir, model_name)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    scenarios = [28, 30, 32]

    for teeth_count in scenarios:
        logger.log(f"Running synthetic scenario with {teeth_count} teeth.")

        tooth_presence = th.ones(32, dtype=th.float32).unsqueeze(0).to(dist_util.dev())

        if teeth_count < 32:
            tooth_presence[0, teeth_count:] = 0.0

        D, H, W = args.image_size // 2, args.image_size // 2, args.image_size // 2

        model_kwargs = {
            "tooth_presence": tooth_presence
        }

        img = th.randn(args.batch_size,
                       8,
                       D,
                       H,
                       W).to(dist_util.dev())

        sample_out = diffusion.p_sample_loop(
            model=model,
            shape=img.shape,
            noise=img,
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

        output_name = os.path.join(output_dir, f"synth_scenario_{teeth_count}_teeth.nii.gz")
        img_nii = nib.Nifti1Image(sample_out[0].detach().cpu().numpy(), np.eye(4))
        nib.save(img_nii, filename=output_name)
        print(f'Saved to {output_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        meta_data="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,
    )

    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
