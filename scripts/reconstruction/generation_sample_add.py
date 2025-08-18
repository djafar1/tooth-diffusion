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
from skimage.morphology import ball
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

def inpaint_teeth(image_np, label_np, tooth_ids, sphere_radius=2):
    struct = ball(sphere_radius)

    tooth_mask = np.zeros_like(label_np, dtype=bool)
    for tooth_id in tooth_ids:
        tooth_mask |= (label_np == tooth_id)

    tooth_mask = binary_dilation(tooth_mask, structure=struct)
    teeth_mask = binary_dilation(label_np > 0, structure=struct)

    V1 = image_np.copy()
    V1[teeth_mask] = np.nan

    missing = np.isnan(V1)
    dist, (inds_z, inds_y, inds_x) = distance_transform_edt(missing, return_indices=True)

    V2 = image_np.copy()
    V2[teeth_mask] = image_np[
        inds_z[teeth_mask],
        inds_y[teeth_mask],
        inds_x[teeth_mask]
    ]

    V2_smooth = gaussian_filter(V2, sigma=1.0)

    inpainted = image_np.copy()
    inpainted[tooth_mask] = V2_smooth[tooth_mask]

    return inpainted

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

    # --- Filter settings ---
    NUM_SAMPLES = args.num_samples      # how many outputs to generate
    MISSING_MIN = 4                     # only process samples with missing in [MIN, MAX]
    MISSING_MAX = 32
    K_ADD = 4                           # try to add back K missing teeth
    # NOTE: K_ADD should NOT exceed the number of missing teeth in a sample.
    # ---------------------------------
    output_dir = os.path.join(args.output_dir, "add_back_missing_by_tooth_presence")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    selected_count = 0

    for sample in loader:
        sample_name = sample["name"][0]
        if isinstance(sample_name, tuple):
            sample_name = sample_name[0]

        # count current missing teeth (before our modification)
        missing_now = (sample["tooth_presence"] == 0).sum().item()

        # filter dataset by missing count
        if not (MISSING_MIN <= missing_now <= MISSING_MAX):
            continue

        # move to device (keep 'name' on CPU)
        base_sample = {}
        for k, v in sample.items():
            if k == "name":
                base_sample[k] = v
            else:
                base_sample[k] = v.to(dist_util.dev())

        raw_name = base_sample["name"][0]
        if isinstance(raw_name, tuple):
            raw_name = raw_name[0]
        base_name = os.path.splitext(os.path.splitext(raw_name)[0])[0]

        base_image = base_sample["image"]            # [1,1,D,H,W]
        base_tp = base_sample["tooth_presence"]      # [1,32]

        logger.log(f"[{selected_count+1}/{NUM_SAMPLES}] Selected: {sample_name}")
        logger.log(f"Sample currently missing {missing_now} teeth (before modification).")

        # indices of currently missing teeth (0-based positions where tp==0)
        missing_indices_0based = (base_tp[0] == 0).nonzero(as_tuple=True)[0].tolist()
        if len(missing_indices_0based) == 0:
            logger.log("No missing teeth to add back; skipping.")
            continue

        # choose up to K_ADD random missing teeth to set to present (1 → add back)
        k = min(K_ADD, len(missing_indices_0based))
        chosen_missing_0based = random.sample(missing_indices_0based, k=k)
        teeth_to_add_back = [i + 1 for i in chosen_missing_0based]  # to 1-based for logging

        logger.log(f"Will try to add back teeth (1-based): {teeth_to_add_back}")

        # modify tooth_presence by setting chosen positions to 1.0
        tooth_presence_modified = base_tp.clone()
        for idx0 in chosen_missing_0based:
            tooth_presence_modified[0, idx0] = 1.0

        # build conditioning from FULL image (no inpainting)
        full_image_th = base_image
        LLL_img, LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img = dwt(full_image_th)
        cond_dwt = th.cat([LLL_img / 3., LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img], dim=1)

        model_kwargs = {"tooth_presence": tooth_presence_modified}

        # start from noise (wavelet space)
        img = th.randn(
            args.batch_size,
            8,
            args.image_size // 2,
            args.image_size // 2,
            args.image_size // 2,
        ).to(dist_util.dev())

        sample_out = diffusion.p_sample_loop(
            t=args.sampling_steps,                          
            model=model,
            shape=img.shape,
            noise=img,
            cond=cond_dwt,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # inverse DWT and save
        B, _, D, H, W = sample_out.size()
        sample_out = idwt(
            sample_out[:, 0, :, :, :].view(B, 1, D, H, W) * 3.0,
            sample_out[:, 1, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 2, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 3, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 4, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 5, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 6, :, :, :].view(B, 1, D, H, W),
            sample_out[:, 7, :, :, :].view(B, 1, D, H, W),
        )

        sample_out = (sample_out + 1) / 2.0
        if len(sample_out.shape) == 5:
            sample_out = sample_out.squeeze(dim=1)

        add_tag = "_".join(map(str, teeth_to_add_back)) if len(teeth_to_add_back) else "none"
        output_name = os.path.join(output_dir, f"tp_added_{add_tag}_{base_name}.nii.gz")

        img_nii = nib.Nifti1Image(sample_out[0].detach().cpu().numpy(), np.eye(4))
        nib.save(img_nii, filename=output_name)
        logger.log(f"Saved: {output_name}")

        selected_count += 1
        if selected_count >= NUM_SAMPLES:
            break

    if selected_count == 0:
        logger.log(f"No samples matched missing range [{MISSING_MIN}, {MISSING_MAX}]; nothing saved.")



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
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
