# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

import numpy as np
import random
import blobfile as bf
import torch
import smplx
from tqdm import tqdm
from torch.optim import AdamW

from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import create_named_schedule_sampler

class TrainLoopTrajNet:
    def __init__(self, args, writer, model, diffusion_train, diffusion_eval, timestep_respacing_eval,
                 start_infill_epoch, max_infill_ratio, mask_prob, train_dataloader, test_dataloader, logdir, logger, device='cpu'):
        self.writer = writer
        self.model = model
        self.diffusion_train = diffusion_train
        self.diffusion_eval = diffusion_eval
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.timestep_respacing_eval = timestep_respacing_eval
        self.start_infill_epoch = start_infill_epoch
        self.mask_prob = mask_prob
        self.max_infill_ratio = max_infill_ratio

        self.smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(device)

        self.step = 0
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,  # MDM
            use_fp16=self.use_fp16,  # False
            fp16_scale_growth=self.fp16_scale_growth,
        )
        self.save_dir = logdir
        self.logger = logger
        self.opt = AdamW(filter(lambda p: p.requires_grad, self.mp_trainer.master_params),
                         lr=self.lr, weight_decay=self.weight_decay)

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion_train)

    def run_loop(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            traj_feat_dim = self.train_dataloader.dataset.traj_feat_dim
            for batch in tqdm(self.train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)

                ######### add occlusion mask for traj repr, with some schedules
                if epoch >= self.start_infill_epoch:
                    prob = random.uniform(0, 1)
                    if prob > 1 - self.mask_prob:
                        clip_len = batch['cond'].shape[1]
                        batch_size = batch['cond'].shape[0]
                        start = torch.FloatTensor(batch_size).uniform_(0, clip_len-1).long()
                        mask_len = (clip_len * torch.FloatTensor(batch_size).uniform_(0, 1) * self.max_infill_ratio).long()
                        end = start + mask_len
                        end[end>clip_len] = clip_len
                        mask_traj = torch.ones(batch_size, clip_len).to(self.device)  # [bs, t]
                        for bs in range(batch_size):
                            mask_traj[bs, start[bs]:end[bs]] = 0
                        mask_traj = mask_traj.unsqueeze(-1).repeat(1, 1, traj_feat_dim)   # [bs, t, 4]
                        batch['cond'][:, :, 0:traj_feat_dim] = batch['cond'][:, :, 0:traj_feat_dim] * mask_traj

                train_losses = self.run_step(batch)

                if self.step % self.log_interval == 0 and self.step > 0:
                    for key in train_losses.keys():
                        self.writer.add_scalar('train/{}'.format(key), train_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. format(self.step, epoch, key, train_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                if self.step % self.log_interval == 0 and self.step > 0:
                    self.model.eval()
                    for test_step, test_batch in tqdm(enumerate(self.test_dataloader)):
                        for key in test_batch.keys():
                            test_batch[key] = test_batch[key].to(self.device)
                        shape = list(test_batch['motion_repr_clean'][:, :, 0:traj_feat_dim].shape)
                        eval_losses, val_output = self.diffusion_eval.eval_losses(model=self.model, batch=test_batch,
                                                                                  shape=shape, progress=False,
                                                                                  clip_denoised=False, cur_epoch=epoch,
                                                                                  timestep_respacing=self.timestep_respacing_eval,
                                                                                  smplx_model=self.smplx_neutral)
                        for key in eval_losses.keys():
                            if test_step == 0:
                                eval_losses[key] = eval_losses[key].detach().clone()
                            if test_step > 0:
                                eval_losses[key] += eval_losses[key].detach().clone()

                    for key in eval_losses.keys():
                        eval_losses[key] = eval_losses[key] / (test_step + 1)
                        self.writer.add_scalar('eval/{}'.format(key), eval_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'.format(self.step, epoch, key, eval_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                    self.model.train()

                if self.step % self.save_interval == 0 and self.step > 0:
                    self.save()

                self.step += 1


    def run_step(self, batch):
        losses = self.forward_backward(batch)
        self.mp_trainer.optimize(self.opt)
        return losses

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(batch['motion_repr_clean'].shape[0], dist_util.dev())
        losses = self.diffusion_train.training_losses(model=self.model, batch=batch, t=t, noise=None,
                                                      traj_feat_dim=self.train_dataloader.dataset.traj_feat_dim,
                                                      smplx_model=self.smplx_neutral)
        loss = (losses["loss"] * weights).mean()
        self.mp_trainer.backward(loss)
        return losses


    def ckpt_file_name(self):
        return f"model{(self.step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)
            self.logger.info('[*] model saved\n')
        save_checkpoint(self.mp_trainer.master_params)

