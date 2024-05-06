# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

import blobfile as bf
import random
from torch.optim import AdamW
import smplx

from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from utils.other_utils import *

class TrainLoopPoseNet:
    def __init__(self, args, writer, model, diffusion_train, diffusion_eval, timestep_respacing_eval, input_noise,
                 train_dataloader, test_dataloader, logdir, logger, start_prox_mask_epoch, mask_scheme, device='cpu'):
        self.args = args
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
        self.input_noise = input_noise
        self.start_prox_mask_epoch = start_prox_mask_epoch
        self.mask_scheme = mask_scheme

        self.smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(device)

        self.step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1
        self.sync_cuda = torch.cuda.is_available()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,  # MDM
            use_fp16=self.use_fp16,  # False
            fp16_scale_growth=self.fp16_scale_growth,
        )
        self.save_dir = logdir
        self.logger = logger
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion_train)


    def run_loop(self):
        ######################################## load prox masks
        print('[INFO] loading PROX joint masks...')
        all_dataset_root = '/'.join(self.args.dataset_root.split('/')[0:-1])
        prox_mask_dir_list = os.listdir(os.path.join(all_dataset_root, 'PROX/mask_joint'))  # ['MPH11_00034_01', 'MPH11_00150_01', ...]
        prox_mask_list = []
        clip_len = self.train_dataloader.dataset.clip_len
        for dir in tqdm(prox_mask_dir_list):
            mask = np.load(os.path.join(all_dataset_root, 'PROX/mask_joint', '{}/mask_joint.npy'.format(dir)))
            n_clip = len(mask) // clip_len
            for i in range(n_clip):
                mask_clip = mask[(i * clip_len):((i + 1) * clip_len)][:, 0:22]  # [T, 22]  1 for visible
                all_joints_n = mask_clip.shape[0] * mask_clip.shape[1]
                mask_joints_n = all_joints_n - mask_clip.sum()
                mask_ratio = mask_joints_n / all_joints_n
                if mask_ratio >= 0.05:  # ignore clips with very few masks
                    cur_mask_vec_vis_dict = {}
                    for key in REPR_LIST:
                        if key in ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel', 'root_height',
                                   'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel', 'smplx_betas']:
                            cur_mask_vec_vis_dict[key] = np.ones((clip_len, REPR_DIM_DICT[key]))
                        elif key in ['local_positions', 'local_vel']:
                            cur_mask_vec_vis_dict[key] = mask_clip[:, 0:].repeat(3, axis=1)
                        elif key == 'smplx_body_pose_6d':
                            cur_mask_vec_vis_dict[key] = mask_clip[:, 1:].repeat(6, axis=1)
                        elif key == 'foot_contact':
                            cur_mask_vec_vis_dict[key] = np.zeros((clip_len, 4))
                            left_foot_visible = (mask_clip[:, 7] == 1) * (mask_clip[:, 10] == 1)  # [T]
                            right_foot_visible = (mask_clip[:, 8] == 1) * (mask_clip[:, 11] == 1)  # [T]
                            cur_mask_vec_vis_dict[key][left_foot_visible, 0:2] = 1.0
                            cur_mask_vec_vis_dict[key][right_foot_visible, 2:] = 1.0
                    cur_mask_vec_vis = np.concatenate([cur_mask_vec_vis_dict[key] for key in cur_mask_vec_vis_dict], axis=-1)  # [T, body_feat_dim]
                    prox_mask_list.append(cur_mask_vec_vis)  # [n_seq, T, body_feat_dim]
        prox_mask_list = np.asarray(prox_mask_list)
        print('[INFO] prox masks loaded, get {} prox mask clips in total.'.format(len(prox_mask_list)))

        ######################################## start training
        for epoch in range(self.num_epochs):
            self.model.train()
            traj_feat_dim = self.train_dataloader.dataset.traj_feat_dim
            for batch in tqdm(self.train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                if not self.input_noise:
                    batch['cond'] = batch['motion_repr_clean'].clone()  # [bs, clip_len, body_feat_dim]
                else:
                    batch['cond'] = batch['motion_repr_noisy'].clone()
                bs, clip_len = batch['motion_repr_clean'].shape[0], batch['motion_repr_clean'].shape[1]

                ####################### add mask, with some schedules
                # mask random 1-6 joints
                if epoch <= self.start_prox_mask_epoch:
                    mask_joint_n = random.randint(1, 6)
                    mask_joint_id = torch.rand(bs, mask_joint_n) * 22  # all 22 joints
                    mask_joint_id = mask_joint_id.long()  # [bs, mask_joint_n]
                    mask_joint_id[mask_joint_id == 0] = 1  # could contain 0 (pelvis), but do not mask out pelvis joint
                    for i in range(bs):
                        for k in range(3):
                            batch['cond'][i, :, traj_feat_dim + mask_joint_id[i] * 3 + k] = 0.
                        for k in range(3):
                            batch['cond'][i, :, traj_feat_dim + 22 * 3 + mask_joint_id[i] * 3 + k] = 0.
                        for k in range(6):
                            batch['cond'][i, :, traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id[i] - 1) * 6 + k] = 0.
                        if 7 in mask_joint_id[i] or 10 in mask_joint_id[i]:  # left foot
                            batch['cond'][i, :, -4:-2] = 0.
                        if 8 in mask_joint_id[i] or 11 in mask_joint_id[i]:  # right foot
                            batch['cond'][i, :, -2:] = 0.
                    if self.input_noise:  # mask out input contact lbl if input condition is noisy
                        batch['cond'][:, :, -4:] = 0.
                # load prox mask and more aggressive masking schemes:
                # lower / upper / full: mask out all lower/upper/full body joints
                else:
                    prob = random.uniform(0, 1)
                    prob_dict = {}
                    if self.mask_scheme == 'lower':
                        prob_dict = {'prox': 0.7,
                                     'lower': 1.0}
                    elif self.mask_scheme == 'lower+upper':
                        prob_dict = {'prox': 0.5,
                                     'lower': 0.8,
                                     'upper': 1.0}
                    elif self.mask_scheme == 'lower+full':
                        prob_dict = {'prox': 0.5,
                                     'lower': 0.8,
                                     'full': 1.0}
                    elif self.mask_scheme == 'lower+upper+full':
                        prob_dict = {'prox': 0.5,
                                     'lower': 0.8,
                                     'upper': 0.9,
                                     'full': 1.0}
                    if 'prox' in prob_dict.keys() and prob <= prob_dict['prox']:
                        np.random.shuffle(prox_mask_list)  # shuffle along n_seq axis
                        prox_mask = torch.from_numpy(prox_mask_list[0:bs]).float().to(self.device)[:, 0:-1]  # [bs, T, body_feat_dim] 1-visible
                        batch['cond'] = batch['cond'] * prox_mask
                    #### mask out lower body part
                    elif 'lower' in prob_dict.keys() and prob <= prob_dict['lower']:
                        lower_joint_id = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])
                        for k in range(3):
                            batch['cond'][:, :, traj_feat_dim + lower_joint_id * 3 + k] = 0.
                        for k in range(3):
                            batch['cond'][:, :, traj_feat_dim + 22 * 3 + lower_joint_id * 3 + k] = 0.
                        for k in range(6):
                            batch['cond'][:, :, traj_feat_dim + 22 * 3 + 22 * 3 + (lower_joint_id - 1) * 6 + k] = 0.
                        batch['cond'][:, :, -4:] = 0.
                    #### mask out upper body part
                    elif 'upper' in prob_dict.keys() and prob <= prob_dict['upper']:
                        upper_joint_id = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                        prob_upper_body_select = random.uniform(0, 1)
                        if prob_upper_body_select < 0.6:
                            upper_joint_id_select = random.sample(upper_joint_id, 5)
                            if 18 not in upper_joint_id_select:
                                upper_joint_id_select = upper_joint_id_select + [18]
                            if 19 not in upper_joint_id_select:
                                upper_joint_id_select = upper_joint_id_select + [19]
                            if 20 not in upper_joint_id_select:
                                upper_joint_id_select = upper_joint_id_select + [20]
                            if 21 not in upper_joint_id_select:
                                upper_joint_id_select = upper_joint_id_select + [21]
                            upper_joint_id_select = np.asarray(sorted(upper_joint_id_select))
                        else:
                            upper_joint_id_select = np.asarray(upper_joint_id)
                        for k in range(3):
                            batch['cond'][:, :, traj_feat_dim + upper_joint_id_select * 3 + k] = 0.
                        for k in range(3):
                            batch['cond'][:, :, traj_feat_dim + 22 * 3 + upper_joint_id_select * 3 + k] = 0.
                        for k in range(6):
                            batch['cond'][:, :, traj_feat_dim + 22 * 3 + 22 * 3 + (upper_joint_id_select - 1) * 6 + k] = 0.
                        batch['cond'][:, :, -4:] = 0.
                    #### mask out full body of a sub-sequence
                    elif 'full' in prob_dict.keys() and prob <= prob_dict['full']:
                        start = torch.FloatTensor(bs).uniform_(0, clip_len - 1).long()  # [bs]
                        mask_len = 30
                        end = start + mask_len
                        end[end > clip_len] = clip_len
                        batch['cond'][:, :, -4:] = 0.
                        for idx in range(bs):
                            batch['cond'][idx, start[idx]:end[idx], traj_feat_dim:] = 0  # do not mask out traj part
                if self.input_noise:
                    batch['cond'][:, :, -4:] = 0.

                batch['motion_repr_clean'] = torch.permute(batch['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
                batch['cond'] = torch.permute(batch['cond'], (0, 2, 1)).unsqueeze(-2)

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
                        if not self.input_noise:
                            test_batch['cond'] = test_batch['motion_repr_clean'].clone()  # [bs, clip_len, 263]
                        else:
                            test_batch['cond'] = test_batch['motion_repr_noisy'].clone()
                        bs, clip_len = test_batch['motion_repr_clean'].shape[0], test_batch['motion_repr_clean'].shape[1]

                        ####################### add mask, mask 1-6 joints randomly
                        mask_joint_n = random.randint(1, 6)
                        mask_joint_id = torch.rand(bs, mask_joint_n) * 22
                        mask_joint_id = mask_joint_id.long()  # [bs, mask_joint_n]
                        mask_joint_id[mask_joint_id == 0] = 1  # do not mask out pelvis joint
                        for i in range(bs):
                            for k in range(3):
                                test_batch['cond'][i, :, traj_feat_dim + mask_joint_id[i] * 3 + k] = 0.
                            for k in range(3):
                                test_batch['cond'][i, :, traj_feat_dim + 22 * 3 + mask_joint_id[i] * 3 + k] = 0.
                            for k in range(6):
                                test_batch['cond'][i, :,
                                traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id[i] - 1) * 6 + k] = 0.
                            if 7 in mask_joint_id[i] or 10 in mask_joint_id[i]:  # left foot
                                test_batch['cond'][i, :, -4:-2] = 0.
                            if 8 in mask_joint_id[i] or 11 in mask_joint_id[i]:  # right foot
                                test_batch['cond'][i, :, -2:] = 0.
                        if self.input_noise:
                            test_batch['cond'][:, :, -4:] = 0.

                        test_batch['motion_repr_clean'] = torch.permute(test_batch['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
                        test_batch['cond'] = torch.permute(test_batch['cond'], (0, 2, 1)).unsqueeze(-2)
                        shape = list(test_batch['motion_repr_clean'].shape)
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
        losses, model_output = self.diffusion_train.training_losses(model=self.model, batch=batch, t=t, noise=None, smplx_model=self.smplx_neutral)
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

