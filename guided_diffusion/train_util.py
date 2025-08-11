import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        target=None,
        training_mode=None,
        conditioning_image=None,
    ):
        self.training_mode=training_mode
        self.target = target
        self.conditioning_image = conditioning_image
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)

        print(self.diffusion.num_timesteps)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')

        self.loss_level = loss_level

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        
        # Get rank and device
        self.rank = dist.get_rank()
        self.device = self.model.device
        self._load_and_sync_parameters()

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
                state_dict = th.load(self.resume_checkpoint, map_location="cpu")
                self.model.module.load_state_dict(state_dict)
        if dist.is_initialized():       
            dist.barrier()
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)
            dist.barrier()
        

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if main_checkpoint:
            checkpoint_name = os.path.basename(main_checkpoint)
            # Because opt is just opt_{checkpoint} hardcoded
            opt_file = f"opt_{checkpoint_name}"
            opt_checkpoint = bf.join(
                bf.dirname(main_checkpoint), opt_file
            )
            if bf.exists(opt_checkpoint):
                if dist.get_rank() == 0 or not dist.is_initialized():
                    logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                    state_dict = th.load(opt_checkpoint, map_location='cpu')
                else: 
                    state_dict = None
                if dist.is_initialized():
                    obj_list = [state_dict]
                    dist.broadcast_object_list(obj_list, src=0)
                    state_dict = obj_list[0]
                if state_dict is not None:
                    self.opt.load_state_dict(state_dict)
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        t = time.time()
        t_start = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            self.datal.sampler.set_epoch(self.step + self.resume_step)
            t_total = time.time() - t
            t = time.time()
            step_start_time = time.time()
            try:
                batch = next(self.iterdatal)
                cond = {}
            except StopIteration:
                if hasattr(self.datal, 'sampler') and hasattr(self.datal.sampler, 'set_epoch'):
                    shuffle_seed = self.step + self.resume_step
                    self.datal.sampler.set_epoch(shuffle_seed)
                    logger.info(f"Rank {self.rank}: Set data shuffle seed to {shuffle_seed}")
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)
                cond = {}
                
            batch = {k: v.to(self.device) for k, v in batch.items()}

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt = self.run_step(batch, cond)

            t_fwd = time.time()-t_fwd

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
            avg_loss = lossmse.detach()
            if dist.is_initialized():
                # Sum losses from all ranks, then divide by world size
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss /= dist.get_world_size()
                
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', avg_loss.item(), global_step=self.step + self.resume_step)            

                if self.step % 200 == 0:
                    image_size = sample_idwt.size()[2]
                    midplane = sample_idwt[0, 0, :, :, image_size // 2]
                    self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                                global_step=self.step + self.resume_step)

                    image_size = sample.size()[2]
                    for ch in range(8):
                        midplane = sample[0, ch, :, :, image_size // 2]
                        self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)

            if self.step % self.log_interval == 0 and (not dist.is_initialized() or self.rank == 0):
                step_elapsed = time.time() - step_start_time
                total_elapsed = time.time() - t_start
                avg_time_per_step = total_elapsed / self.step
                print(
                    f"[Step {self.step}] Step time: {step_elapsed:.2f}s | "
                    f"Avg per step: {avg_time_per_step:.2f}s | "
                    f"Total elapsed: {total_elapsed/60:.1f} min"
                )
                logger.dumpkvs()
            
            if self.step % 100 == 0:
                logger.logkv(f"rank_{self.rank}_step", self.step)
                logger.logkv(f"rank_{self.rank}_samples", (self.step + self.resume_step) * self.batch_size)
                print(f"[Rank {self.rank}] Step {self.step} — processed {(self.step + self.resume_step) * self.batch_size} samples.")

            if self.step % self.save_interval == 0 and self.rank==0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if self.rank == 0:
                self.save()
            if dist.is_initialized():
                dist.barrier()

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(f"Rank {self.rank}: Model parameters non-finite {param_max_norm}", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Rank {self.rank}: Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None
        
        target_img = 'image'  

        if self.conditioning_image != "none":
            condition_image = 'cond_image'
        else:
            condition_image = None
        for i in range(0, batch[target_img].shape[0], self.microbatch):
            micro_target = batch[target_img][i: i + self.microbatch].to(self.device)
            if condition_image is not None: 
                micro_condition = batch[condition_image][i: i + self.microbatch].to(self.device)
            else:
                micro_condition = None
            
            micro_tooth_presence = batch['tooth_presence'][i: i + self.microbatch].to(self.device)
            micro_label = batch['label'][i: i + self.microbatch].to(self.device) # The mask label for teeth used for loss
            if cond is not None:
                micro_cond = {k: v[i: i + self.microbatch].to(self.device) for k, v in cond.items()}
            else:
                micro_cond = {}
            
            micro_cond['tooth_presence'] = micro_tooth_presence
            micro_cond['condition'] = micro_condition # Conditioning image

            last_batch = (i + self.microbatch) >= batch[target_img].shape[0]
            t, weights = self.schedule_sampler.sample(micro_target.shape[0], self.device)
            
            compute_losses = functools.partial(self.diffusion.training_losses,
                                               self.model,
                                               x_start=micro_target,
                                               t=t,
                                               model_kwargs=micro_cond,
                                               labels=micro_label,
                                               mode=self.mode,
                                               )
            losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1["loss"].detach()
                )

            losses = losses1[0]         # Loss value
            sample = losses1[1]         # Denoised subbands at t=0
            sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0


            # We have to aggregates losses across devices
            mse_wav = losses["mse_wav"].clone().detach()
            dist.all_reduce(mse_wav)
            mse_wav.div_(dist.get_world_size())
            if "masked_mse" in losses:
                masked_mse = losses["masked_mse"].clone().detach()
                dist.all_reduce(masked_mse)
                masked_mse.div_(dist.get_world_size())     
            else:
                masked_mse = None       


            weights = th.ones(len(losses["mse_wav"]), device=self.device)# Equally weight all wavelet channel losses
            
            if self.rank == 0:
                # Log wavelet level loss
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('loss/mse_wav_lll', mse_wav[0].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_llh', mse_wav[1].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_lhl', mse_wav[2].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_lhh', mse_wav[3].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hll', mse_wav[4].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hlh', mse_wav[5].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hhl', mse_wav[6].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hhh', mse_wav[7].item(),
                                                global_step=self.step + self.resume_step)
                if masked_mse is not None:
                    self.summary_writer.add_scalar('loss/masked_mse', masked_mse.item(), global_step=self.step + self.resume_step)
                    logger.logkv_mean("masked_mse", masked_mse.item())
                log_loss_dict(self.diffusion, t, {"mse_wav": mse_wav * weights.to(self.device)})
                
            loss = (losses["mse_wav"] * weights).mean()
        
            if "masked_mse" in losses:
                lambda_mask = 10.0  # adjusted from 1.0 to 10.
                loss = loss + lambda_mask * losses["masked_mse"]
            
            lossmse = loss.detach()
            
            # perform some finiteness checks
            if not torch.isfinite(loss):
                logger.log(f"Rank {self.rank}: Encountered non-finite loss {loss}")
            if self.use_fp16:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
                
            return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.logkv("step", self.step + self.resume_step)
            logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log("Saving model...")
                if self.dataset == 'brats':
                    filename = f"brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'my_data':
                    if self.training_mode == "bdtrain":
                        filename = f"my_data_bd_{(self.step+self.resume_step):06d}.pt"
                    else: # if just regular training
                        if self.conditioning_image == "brain" or self.conditioning_image == "skull":
                            condition_str = self.conditioning_image
                        else:
                            condition_str = "none"
                        filename = f"my_data_t_{self.target}_c_{condition_str}_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'tooth':
                    cond_str = "none"
                    if self.conditioning_image is not None and self.conditioning_image != "none":
                        cond_str = "tooth"
                    filename = f"tooth_target_tooth_cond_{cond_str}_{(self.step+self.resume_step):06d}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())


        #The opt is hardcoded to my_data now, can be adjusted if name of dataset changes, or expanded if necesarry.
        if not dist.is_initialized() or dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            if self.dataset == "tooth":
                cond_str = "none"
                if self.conditioning_image is not None and self.conditioning_image != "none":
                    cond_str = "tooth"
                optfilename = f"opt_tooth_target_tooth_cond_{cond_str}_{(self.step + self.resume_step):06d}.pt"
                
                
            elif self.training_mode == "bdtrain":
                optfilename = f"opt_my_data_bd_{(self.step + self.resume_step):06d}.pt"
            else:
                if self.conditioning_image == "brain" or self.conditioning_image == "skull":
                    condition_str = self.conditioning_image
                else:
                    condition_str = "none"
                optfilename = f"opt_my_data_t_{self.target}_c_{condition_str}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(
                bf.join(checkpoint_dir, optfilename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
