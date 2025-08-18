# Wavelet Tooth Diffusion: Guided 3D CBCT Synthesis

This repository contains the code for **Wavelet Tooth Diffusion**, a 3D conditional diffusion framework for dental CBCT imaging. The model integrates **wavelet-domain denoising diffusion** with **tooth-level conditioning** to enable realistic synthesis, reconstruction, and editing of CBCT scans.

## Abstract
Despite the growing importance of dental CBCT scans for diagnosis and treatment planning, generating anatomically realistic scans 
with fine-grained control remains a challenge in medical image synthesis. In this work, we propose a novel conditional diffusion framework
for 3D dental volume generation, guided by tooth-level binary attributes that allow precise control over tooth presence and configuration. Our
approach integrates wavelet-based denoising diffusion, FiLM conditioning, and masked loss functions to focus learning on relevant anatomical
structures. We evaluate the model across diverse tasks, such as tooth addition, removal, and full dentition synthesis, using both paired and
distributional similarity metrics. Results show strong fidelity and generalization with low FID scores, robust inpainting performance, and SSIM
values above 0.91 even on unseen scans. These findings highlight the potential of controllable dental image synthesis for data augmentation,
simulation, and clinically informed AI applications.

![Wavelet Architecture](./assets/wdm.png)

## Highlights

- **Wavelet-based diffusion** for efficient and high-resolution 3D CBCT volume generation.  
- **Fine-grained tooth conditioning** via presence vectors (32 teeth).  
- Supports multiple modes:
  - **Reconstruction** of CBCT volumes  
  - **Tooth addition** (inpainting missing teeth)  
  - **Tooth removal** (simulating absence for augmentation)  
- **Masked loss functions** to focus learning on tooth-bearing regions.  
- **Distributed training** with PyTorch DDP.

## Dependencies

Create and activate the environment with:

```bash
mamba env create -f environment.yml
mamba activate wdm
```

## Data Setup

Organize CBCT data as:

```
prep_data/
  train/
    Images/
    Labels/
  valid/
    Images/
    Labels/
  test/
    Images/
    Labels/
  MetaData.xlsx
```

- **Images/** and **Labels/** contain paired 3D CBCT volumes (`.nii.gz`).  
- **MetaData.xlsx** encodes patient information and tooth presence (1–32). 

## Usage

All training and sampling is handled via `run.sh`.


### Training

```bash
bash run.sh train teeth "" [teeth|""]
```

- `train` – training mode  
- `teeth` – task target  
- `""` – train from scratch (replace with checkpoint path to resume)  
- `[teeth|"" ]` – last argument can be either:  
  - `teeth` → enable tooth conditioning  
  - `""` (left empty) → no conditioning (synthetic generation)  

---

### Sampling

```bash
bash run.sh sample teeth ./checkpoints/model.pt [teeth|""]
```

- `sample` – sampling mode  
- `teeth` – task target  
- `./checkpoints/model.pt` – path to trained checkpoint  
- `[teeth|"" ]` – last argument can be either:  
  - `teeth` → enable tooth conditioning  
  - `""` (left empty) → no conditioning (synthetic generation)  

Generated CBCT volumes are saved to `./results/`.

## Util

The `util/` folder contains helper scripts for:

- `preproc_tooth_data.py`: Uses bounding box detection, clipping, and padding to fit CBCT volumes into a `256³` space.

## Scripts

Sampling scripts are divided into two folders:

### Reconstruction (`scripts/reconstruction/`):
- `generation_sample_add.py`
- `generation_sample_inpaint_all.py`
- `generation_sample_remove.py`

### Synthetic (`scripts/synthetic/`):
- `generation_sample_synth.py`

> **Note:** In `run.sh`, you need to manually update the line:
> 
> ```bash
> python scripts/reconstruction/generation_sample_add.py $SAMPLE $COMMON
> ```
>
> depending on which script you want to use (e.g., replace with `generation_sample_remove.py`, etc).

## Acknowledgements

Our code is based on / inspired by the following repositories:

* [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion) (MIT License)
* [https://github.com/pfriedri/wdm-3d](https://github.com/pfriedri/wdm-3d) (MIT License)
* [https://github.com/pfriedri/cwdm](https://github.com/pfriedri/cwdm) (MIT License)

Thanks to the authors for making these resources openly available.

---
