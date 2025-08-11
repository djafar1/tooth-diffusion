# WDM-3D-COND-V2

This repository contains the final implementation for the Bachelor's thesis: **Brain MRI Generation with Guidance and Explainability**, conducted at the University of Copenhagen, Department of Computer Science.

The project extends the WDM-3D diffusion model for 3D brain MRI synthesis by enabling **conditional generation**, **explainability**, **fairness evaluation**, and **counterfactual synthesis** through the design of an advanced model called `wdm-3d-cond-v2`.

## Dependencies

Create and activate the environment with:

```bash
mamba env create -f environment.yml
mamba activate wdm
```

This version builds directly on top of the WDM-3D architecture, integrating:

* Demographic variable conditioning (age, sex, diagnosis)
* Optional image conditioning (e.g., `brain`, `skull`, or `none`)
* Flexible generation target specification (`brain` or `skull`)

All evaluations and final experiments in the thesis are performed using this version.

## Data Structure

The input data is structured as follows:

```
data/
  metadata.csv         # Contains demographic variables for all subjects
  train/
    brain/
    images/
    labels/
    mask/
  valid/
    brain/
    images/
    labels/
    mask/
  test/
    brain/
    images/
    labels/
    mask/
```

### Metadata

* `metadata.csv` contains columns such as `subject_id`, `age`, `sex`, and `diagnosis`, and must reside in the top-level `data/` directory.

## Final Architecture

The model used is a **WavU-Net-based Wavelet Diffusion Model** with demographic conditioning embedded into the timestep and optional image-based conditioning via DWT-concatenation.

![WDM-3D-COND-V2 Architecture](../wunet-cond.png)

## Models and Evaluation

We explore two conditional generation models:

### Approach 1: Brain Synthesis from Noise

* Goal: Explore **fairness** and **explainability** by generating brain MRIs from pure noise.
* Conditioning: Demographic variables only.
* Evaluation: FID, SSIM, BRISQUE, PSNR, and biomarker alignment (volume, surface area).

### Approach 2: Brain Reconstruction from Brain

* Goal: Explore **counterfactual synthesis** by reconstructing a brain from itself while altering the demographic conditioning.
* Conditioning: Both brain image and demographic variables.
* Evaluation: Wilcoxon test, 1-Wasserstein distance, heatmap analysis of structural differences.

Both models were trained using NVIDIA A100 (40GB and 80GB) GPUs.
## Running the Code

All training and sampling is managed through `run.sh`, which supports training from scratch, resuming from checkpoints, and sampling with DDPM.

### Required Arguments:

```bash
bash run.sh <mode> <target> <checkpoint> <conditioning_image>
```

Where:

* `mode`: `train`, `bdtrain`, or `sample`
* `target`: `brain` or `skull`
* `checkpoint`: Path to a `.pt` file or empty string
* `conditioning_image`: `brain`, `skull`, or `none`

### Example: Training

```bash
bash run.sh train brain "" brain
```

### Example: Sampling

```bash
bash run.sh sample brain ./checkpoints/my_model.pt none
```

## Util

The `util/` folder contains helper scripts for:

* `brain_pullout.py`: Extracts brain volumes from labeled MRI scans by removing background (`labels==0`) and generates binary brain masks
* `data_splitter.py`: Splits the full dataset into structured `train`, `valid`, and `test` sets based on configurable ratios

## Scripts

* `generation_train.py` — Main training loop for the wdm-3d-cond-v2 model
* `generation_sample_brain_approach2.py` — Counterfactual synthesis evaluation (Approach 2)
* `generation_sample_random_cond.py` — Demographic-based sampling for fairness evaluation (Approach 1)
* `generation_sample_brain_random.py` — Reconstructs randomly selected brain images using their own associated demographic and image conditions
* `generation_sample_30.py` — For Approach 1, uses `group_csv=./train_data_metrics.csv` in `run.sh` to generate a matching number of samples for each of the 30 demographic groups
* `generation_sample_30_explain.py`, `generation_sample_20_explain.py` — Explainability evaluations (Approach 1)
* `generation_sample_cond.py` — Baseline conditional generation

---

## Acknowledgements

Our code is based on / inspired by the following repositories:

* [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion) (MIT License)
* [https://github.com/pfriedri/wdm-3d](https://github.com/pfriedri/wdm-3d) (MIT License)
* [https://github.com/pfriedri/cwdm](https://github.com/pfriedri/cwdm) (MIT License)

Thanks to the authors for making these resources openly available.

---

This implementation is self-contained and intended to be used directly on structured ADNI-derived data, with segmentation labels obtained via FreeSurfer preprocessing. All modeling, training, evaluation, and sampling tasks are conducted using `wdm-3d-cond-v2` only.

This work was conducted as part of a Bachelor's thesis project at the University of Copenhagen.
