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


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def compute_gradient_mag(volume: th.Tensor) -> th.Tensor:
    dwt = DWT_3D("haar")
    with th.no_grad():
        LLL, *_ = dwt(volume) 
    LLL_np = LLL.squeeze().cpu().numpy()

    # Compute gradients
    grad_x = ndimage.sobel(LLL_np, axis=0)
    grad_y = ndimage.sobel(LLL_np, axis=1)
    grad_z = ndimage.sobel(LLL_np, axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Normalize to [-1, 1]
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
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
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

    base_sample = None
    
    for sample in loader:
        missing = (sample['tooth_presence'] == 0).sum().item()
        if missing == 3:
            base_sample = sample
            logger.log(f"Selected sample: {sample['name'][0]} with {missing} missing teeth.")
            break

    if base_sample is None:
        raise RuntimeError("No suitable sample with ≥3 missing teeth found.")

    base_sample = {}
    for k, v in sample.items():
        if k == "name":
            base_sample[k] = v
        else:
            base_sample[k] = v.to(dist_util.dev())

    raw_name = base_sample['name'][0]  # e.g., '1001162439_20140520.nii.gz'
    if isinstance(raw_name, tuple):    # in case it's still a tuple somehow
        raw_name = raw_name[0]
    base_name = os.path.splitext(os.path.splitext(raw_name)[0])[0]  # removes .nii.gz
        
    base_image = base_sample['image']
    base_tp = base_sample['tooth_presence']
    base_crown_fill = base_sample['crown_fill']
    base_root_crown = base_sample['root_crown']
    base_bridge = base_sample['bridge']
    base_implant = base_sample['implant']

    scenarios = ["regular", "add", "remove"]
    
    
    for scenario in scenarios:
        logger.log(f"Running scenario: {scenario}")

        tooth_presence = base_tp.clone()
        crown_fill = base_crown_fill.clone()
        root_crown = base_root_crown.clone()
        bridge = base_bridge.clone()
        implant = base_implant.clone()
        
        present_indices = (tooth_presence[0] > 0).nonzero(as_tuple=True)[0].tolist()
        missing_indices = (tooth_presence[0] == 0).nonzero(as_tuple=True)[0].tolist()

        if scenario == "add" and missing_indices:
            num = np.random.randint(1, max(2, len(missing_indices)//4)+1)
            selected = np.random.choice(missing_indices, num, replace=False)
            logger.log(f"Adding teeth: {selected}")
            for idx in selected:
                tooth_presence[0, idx] = 1.0

        elif scenario == "remove" and present_indices:
            num = np.random.randint(1, max(2, len(present_indices)//4)+1)
            selected = np.random.choice(present_indices, num, replace=False)
            logger.log(f"Removing teeth: {selected}")
            for idx in selected:
                tooth_presence[0, idx] = 0.0
                crown_fill[0, idx] = 0.0
                root_crown[0, idx] = 0.0
                bridge[0, idx] = 0.0
                implant[0, idx] = 0.0

        LLL_gradient_mag = compute_gradient_mag(base_image).to(dist_util.dev())
        
        # DWT on base image
        LLL_img, LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img = dwt(base_image)
        
        # Concat LLL gradient magnitude with DWT coefficients of base image
        cond_dwt = th.cat([LLL_gradient_mag, LLL_img / 3., LLH_img, LHL_img, LHH_img, HLL_img, HLH_img, HHL_img, HHH_img], dim=1)
        
        model_kwargs = {
            "tooth_presence": tooth_presence,
            "crown_fill": crown_fill,
            "root_crown": root_crown,
            "bridge": bridge,
            "implant": implant,
        }
        
        img = th.randn(args.batch_size,         # Batch size
                8,                       # 8 wavelet coefficients
                args.image_size//2,      # Half spatial resolution (D)
                args.image_size//2,      # Half spatial resolution (H)
                args.image_size//2,      # Half spatial resolution (W)
                ).to(dist_util.dev())

        sample_out = diffusion.p_sample_loop(
                        model=model,
                        shape=img.shape,
                        noise=img,
                        cond=cond_dwt,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,)
        
        B, _, D, H, W = sample_out.size()

        sample_out = idwt(sample_out[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample_out[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample_out[:, 7, :, :, :].view(B, 1, D, H, W))

        sample_out = (sample_out + 1) / 2.

        if len(sample_out.shape) == 5:
            sample_out = sample_out.squeeze(dim=1)  # don't squeeze batch dimension for bs 1
            
        model_name = os.path.basename(args.model_path).replace('.pt', '')
        output_dir = os.path.join(args.output_dir, model_name)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(sample_out.shape[0]):
            output_name = os.path.join(output_dir,f"sample_{base_name}_{scenario}.nii.gz")
            img_nii = nib.Nifti1Image(sample_out[0].detach().cpu().numpy(), np.eye(4))
            nib.save(img_nii, filename=output_name)
            print(f'Saved to {output_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        meta_data="",
        atlas_label_path="",
        group_csv="",
        slice_start=0,
        slice_end=None,
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
