"""
A script for training a diffusion model to unconditional image generation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import logger
from guided_diffusion.toothloader import ToothVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    th.cuda.set_device(local_rank)
    device = th.device(f'cuda:{local_rank}')
    
    summary_writer = None
    if args.use_tensorboard and rank == 0:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    logger.log(f"Rank {rank}/{world_size}: Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    
    # Model and diffusion creation
    model, diffusion = create_model_and_diffusion(**arguments)
    model = model.to(device)
    model = th.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) # Wrap model for distributed training

    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    logger.log(f"Rank {rank}: Creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    if args.dataset == 'tooth':
        assert args.image_size in [256], "We currently just support image sizes 256"
        ds = ToothVolumes(
            directory=args.data_dir,
            metadata_path=args.meta_data,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode='train',
            img_size=args.image_size,
            augment_missing_teeth=args.augment_missing_teeth,
            reconstruct_3_mode=args.reconstruct_3_mode,
        )

    else:
        print("We currently just support the datasets: tooth")

    logger.log(f"Rank {rank}: Creating dataset...")
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    datal = DataLoader(ds,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False,
                        sampler=sampler,
                        pin_memory=True,
                        drop_last=True,
                        )

    logger.log(f"Rank {rank}: Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='default',
        target=args.target,
        training_mode=args.training_mode,
        conditioning_image=args.conditioning_image,
        lambda_mask=args.lambda_mask,
    ).run_loop()
    
    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        meta_data="",
        target="",
        augment_missing_teeth=False,
        reconstruct_3_mode=False,
        training_mode="train",
        conditioning_image="none",
        schedule_sampler="uniform",
        lr=1e-4,
        lambda_mask=10.0,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        beta_min=0.1,
        beta_max=20.0,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='tooth',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
