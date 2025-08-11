# general settings
GPU=2;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
DATASET='tooth';          
MODEL='ours_wnet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'
MODE=${1:-train}          # train vs sample
TARGET=${2:-teeth}        
RESUME_CHECKPOINT=${3:-}  
CONDITIONING_IMAGE=${4:-none}

# Settings for training 
AUGMENT_MISSING=False; # Whether to augment missing teeth for training done for model without conditioning image
RECONSTRUCT_3_MODE=True; # For training the model with 3 different scenarios, regular, add and removal of teeth done for model with conditioning image

# settings for sampling/inference
ITERATIONS=1200;        # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0        # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="";             # tensorboard dir to be set for the evaluation

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
  IN_CHANNELS=8;  
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

#Change number of channels depending on conditioning image or not. Default 8, so no else statment
# if conditioning image is brain or skull, then 16
if [[ $CONDITIONING_IMAGE != 'none' ]]; then
  IN_CHANNELS=16
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1;
  DATA_DIR=../prep_data/train;
  META_DATA=../prep_data/MetaData.xlsx
elif [[ $MODE == 'train']]; then
  if [[ $DATASET == 'tooth' ]]; then
    echo "MODE: training";
    echo "DATASET: TOOTH"
    DATA_DIR=../prep_data/train;
    META_DATA=../prep_data/MetaData.xlsx
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--timestep_respacing=
--learn_sigma=False
--use_scale_shift_norm=True
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
"
TRAIN="
--data_dir=${DATA_DIR}
--meta_data=${META_DATA}
--target=${TARGET}
--augment_missing_teeth=${AUGMENT_MISSING}
--reconstruct_3_mode=${RECONSTRUCT_3_MODE}
--training_mode=${MODE}
--conditioning_image=${CONDITIONING_IMAGE}
--resume_checkpoint=${RESUME_CHECKPOINT}
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=50000
--num_workers=10
--devices=${GPU}
"
SAMPLE="
--data_dir=${DATA_DIR}
--meta_data=${META_DATA}
--group_csv=${GROUP_CSV}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=${RESUME_CHECKPOINT}
--devices=${GPU}
--output_dir=./results/
--num_samples=100
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"

# run the python scripts
if [[ $MODE == 'train' || $MODE == 'bdtrain' ]]; then
  echo "Training mode: $MODE";
  echo "Target: $TARGET";
  echo "Condition image: $CONDITIONING_IMAGE";
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=12345 scripts/generation_train.py $TRAIN $COMMON
else
  python scripts/generation_sample_brain_approac.py $SAMPLE $COMMON;
fi
