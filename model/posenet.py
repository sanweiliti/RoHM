# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

from utils.other_utils import *
import smplx
from data_loaders.motion_representation import recover_from_repr_smpl
from model.heads import *



class PoseNet(nn.Module):
    def __init__(self, dataset, body_feat_dim, nfeats=1,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 body_model_path='',
                 device=None,
                 traj_feat_dim=4,
                 weight_loss_rec_repr_full_body=0.0,
                 weight_loss_repr_foot_contact_mse=0.0,
                 weight_loss_joint_pos_global=0.0,
                 weight_loss_joint_vel_global=0.0, weight_loss_joint_smooth=0.0,
                 weight_loss_foot_skating=0.0,
                 start_skating_loss_epoch=0,
                 ):
        super().__init__()
        self.dataset = dataset
        self.body_feat_dim = body_feat_dim
        self.nfeats = nfeats  # 1
        self.traj_feat_dim = traj_feat_dim

        # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
        self.foot_joint_index_list = [7, 10, 8, 11]
        self.foot_skating_vel_thres = 0.1
        self.fps = 30

        self.latent_dim = latent_dim  # 512
        self.ff_size = ff_size  # 1024
        self.num_layers = num_layers  # 8
        self.num_heads = num_heads  # 4
        self.dropout = dropout  # 0.1
        self.activation = activation
        self.input_feats = self.body_feat_dim * self.nfeats
        self.normalize_output = False

        self.mse_loss = nn.MSELoss(reduction='none').to(device)
        self.l1_loss = nn.L1Loss(reduction='none').to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.device = device

        self.weight_loss_rec_repr_full_body = weight_loss_rec_repr_full_body
        self.weight_loss_repr_foot_contact_mse = weight_loss_repr_foot_contact_mse
        self.weight_loss_joint_pos_global = weight_loss_joint_pos_global
        self.weight_loss_joint_vel_global = weight_loss_joint_vel_global
        self.weight_loss_joint_smooth = weight_loss_joint_smooth
        self.weight_loss_foot_skating = weight_loss_foot_skating
        self.start_skating_loss_epoch = start_skating_loss_epoch

        self.smplx_model = smplx.create(model_path=body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.input_process_cond = InputProcess(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.output_process = OutputProcess(self.dataset.pose_feat_dim, self.latent_dim, self.nfeats)


    def forward(self, batch, timesteps):
        """
        Input:
            batch['x_t']: [bs, body_feat_dim, 1, T]
            timesteps: [bs] (int)
        Output:
            output: [bs, pose_feat_dim, 1, T]
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, 512]

        x = self.input_process(batch['x_t'])
        cond = self.input_process_cond(batch['cond'])
        x = x + cond

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [T+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [T+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # [T, bs, d]
        output = self.output_process(output)  # [bs, pose_feat_dim, 1, T]
        given_traj = batch['cond'][:, 0:self.traj_feat_dim, :]  # gt traj for training, trajnet output for test
        output = torch.cat([given_traj, output], dim=1)  # [bs, pose_feat_dim, 1, T]
        return output


    def compute_losses_with_smpl(self, batch, model_output, smplx_model=None, epoch=0):
        # model_output: [bs, pose_feat_dim, 1, T]
        loss_dict = {}

        ###################### loss on full body repr
        loss_rec_repr_all = self.mse_loss(batch['motion_repr_clean'], model_output)
        loss_rec_repr_all = loss_rec_repr_all[:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        loss_dict['loss_repr_full_body'] = loss_rec_repr_all[:, :, self.traj_feat_dim:-4].mean()

        ###################### loss on global joint coordinate
        full_repr_clean = batch['motion_repr_clean'][:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        full_repr_clean = full_repr_clean * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = full_repr_clean[..., cur_total_dim:(cur_total_dim+REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        joint_pos_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='joint_abs_traj', smplx_model=smplx_model)

        full_repr_rec = model_output[:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        full_repr_rec = full_repr_rec * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_rec[repr_name] = full_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        joint_pos_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj', smplx_model=smplx_model)  # [bs, clip_len, 22, 3]
        joint_pos_rec_from_rel_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_rel_traj', smplx_model=smplx_model)
        joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_model)
        loss_dict['loss_joint_pos_global_from_abs_traj'] = self.mse_loss(joint_pos_rec_from_abs_traj, joint_pos_clean).mean()
        loss_dict['loss_joint_pos_global_from_rel_traj'] = self.mse_loss(joint_pos_rec_from_rel_traj, joint_pos_clean).mean()
        loss_dict['loss_joint_pos_global_from_smpl'] = self.mse_loss(joint_pos_rec_from_smpl, joint_pos_clean).mean()

        ###################### loss on global joint velocity
        joint_vel_clean = joint_pos_clean[:, 1:] - joint_pos_clean[:, 0:-1]
        joint_vel_rec_from_abs_traj = joint_pos_rec_from_abs_traj[:, 1:] - joint_pos_rec_from_abs_traj[:, 0:-1]  # [bs, clip_len-1, 22, 3]
        joint_vel_rec_from_rel_traj = joint_pos_rec_from_rel_traj[:, 1:] - joint_pos_rec_from_rel_traj[:, 0:-1]
        joint_vel_rec_from_smpl = joint_pos_rec_from_smpl[:, 1:] - joint_pos_rec_from_smpl[:, 0:-1]
        loss_dict['loss_joint_vel_global_from_abs_traj'] = self.mse_loss(joint_vel_rec_from_abs_traj, joint_vel_clean).mean()
        loss_dict['loss_joint_vel_global_from_rel_traj'] = self.mse_loss(joint_vel_rec_from_rel_traj, joint_vel_clean).mean()
        loss_dict['loss_joint_vel_global_from_smpl'] = self.mse_loss(joint_vel_rec_from_smpl, joint_vel_clean).mean()

        ###################### accel smooth regularizor
        joint_acc_rec_from_abs_traj = joint_vel_rec_from_abs_traj[:, 1:] - joint_vel_rec_from_abs_traj[:, 0:-1]
        joint_acc_rec_from_rel_traj = joint_vel_rec_from_rel_traj[:, 1:] - joint_vel_rec_from_rel_traj[:, 0:-1]
        joint_acc_rec_from_smpl = joint_vel_rec_from_smpl[:, 1:] - joint_vel_rec_from_smpl[:, 0:-1]
        loss_dict['loss_joint_smooth_from_abs_traj'] = torch.mean(joint_acc_rec_from_abs_traj ** 2)
        loss_dict['loss_joint_smooth_from_rel_traj'] = torch.mean(joint_acc_rec_from_rel_traj ** 2)
        loss_dict['loss_joint_smooth_from_smpl'] = torch.mean(joint_acc_rec_from_smpl ** 2)

        ###################### contact lbl loss
        loss_dict['loss_repr_foot_contact_mse'] = self.mse_loss(batch['motion_repr_clean'][:, -4:, :, :], model_output[:, -4:, :, :]).mean()

        ###################### foot skating loss
        contact_lbl_gt = full_repr_clean[:, :, -4:]  # [bs, clip_len, 4] 1 - in contact, 0 - not in contact

        foot_joint_rec_vel_from_abs_traj = (joint_pos_rec_from_abs_traj[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_abs_traj[:, 0:-1, self.foot_joint_index_list]) * self.fps  # [bs, clip_len-1, 4, 3]
        foot_joint_rec_vel_from_abs_traj = torch.norm(foot_joint_rec_vel_from_abs_traj, dim=-1)  # [bs, clip_len-1, 4]
        mask_skating_from_abs_traj = (foot_joint_rec_vel_from_abs_traj - self.foot_skating_vel_thres).gt(0)  # [bs, clip_len-1, 4]  True/False
        mask_skating_from_abs_traj = mask_skating_from_abs_traj * contact_lbl_gt[:, 0:-1]  # [bs, clip_len-1, 4]
        masked_foot_joint_rec_vel_from_abs_traj = foot_joint_rec_vel_from_abs_traj * mask_skating_from_abs_traj  # [bs, clip_len-1, 4]
        loss_dict['loss_foot_skating_from_abs_traj'] = masked_foot_joint_rec_vel_from_abs_traj.sum() / mask_skating_from_abs_traj.sum()

        foot_joint_rec_vel_from_rel_traj = (joint_pos_rec_from_rel_traj[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_rel_traj[:, 0:-1, self.foot_joint_index_list]) * self.fps
        foot_joint_rec_vel_from_rel_traj = torch.norm(foot_joint_rec_vel_from_rel_traj, dim=-1)
        mask_skating_from_rel_traj = (foot_joint_rec_vel_from_rel_traj - self.foot_skating_vel_thres).gt(0)
        mask_skating_from_rel_traj = mask_skating_from_rel_traj * contact_lbl_gt[:, 0:-1]
        masked_foot_joint_rec_vel_from_rel_traj = foot_joint_rec_vel_from_rel_traj * mask_skating_from_rel_traj
        loss_dict['loss_foot_skating_from_rel_traj'] = masked_foot_joint_rec_vel_from_rel_traj.sum() / mask_skating_from_rel_traj.sum()

        foot_joint_rec_vel_from_smpl = (joint_pos_rec_from_smpl[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_smpl[:, 0:-1, self.foot_joint_index_list]) * self.fps
        foot_joint_rec_vel_from_smpl = torch.norm(foot_joint_rec_vel_from_smpl, dim=-1)
        mask_skating_from_smpl = (foot_joint_rec_vel_from_smpl - self.foot_skating_vel_thres).gt(0)
        mask_skating_from_smpl = mask_skating_from_smpl * contact_lbl_gt[:, 0:-1]
        masked_foot_joint_rec_vel_from_smpl = foot_joint_rec_vel_from_smpl * mask_skating_from_smpl
        loss_dict['loss_foot_skating_from_smpl'] = masked_foot_joint_rec_vel_from_smpl.sum() / mask_skating_from_smpl.sum()

        if epoch >= self.start_skating_loss_epoch:
            weight_loss_foot_skating = self.weight_loss_foot_skating
        else:
            weight_loss_foot_skating = 0.0


        loss_dict["loss"] = self.weight_loss_rec_repr_full_body * loss_dict['loss_repr_full_body'] + \
                            self.weight_loss_repr_foot_contact_mse * loss_dict['loss_repr_foot_contact_mse'] + \
                            self.weight_loss_joint_pos_global * (loss_dict['loss_joint_pos_global_from_abs_traj'] + loss_dict['loss_joint_pos_global_from_rel_traj'] + loss_dict['loss_joint_pos_global_from_smpl']) + \
                            self.weight_loss_joint_vel_global * (loss_dict['loss_joint_vel_global_from_abs_traj'] + loss_dict['loss_joint_vel_global_from_rel_traj'] + loss_dict['loss_joint_vel_global_from_smpl'])+ \
                            self.weight_loss_joint_smooth * (loss_dict['loss_joint_smooth_from_abs_traj'] + loss_dict['loss_joint_smooth_from_rel_traj'] + loss_dict['loss_joint_smooth_from_smpl']) + \
                            weight_loss_foot_skating * (loss_dict['loss_foot_skating_from_abs_traj'] + loss_dict['loss_foot_skating_from_rel_traj'] + loss_dict['loss_foot_skating_from_smpl'])
        return loss_dict


    def guide_skating_with_smpl(self, batch, out, denoise_t, compute_grad='x_t'):
        # model_output: [bs, pose_feat_dim, 1, T]
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, body_feat_dim, 1, T]

            ########## obtain global joint coordinate
            full_repr_rec = x_t[:, :, 0].permute(0, 2, 1)  # [bs, T, body_feat_dim]
            full_repr_rec = full_repr_rec * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)

            # reconstruct joint positions
            traj_feat_dim = self.dataset.traj_feat_dim
            cur_total_dim = 0
            repr_dict_rec = {}
            for repr_name in REPR_LIST:
                repr_dict_rec[repr_name] = full_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
                cur_total_dim += REPR_DIM_DICT[repr_name]
            joint_pos_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj', smplx_model=self.smplx_model)  # [bs, clip_len, 22, 3]
            joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=self.smplx_model)

            ############# foot skating loss
            contact_lbl_rec = full_repr_rec[:, :, -4:].detach().clone()  # [bs, clip_len, 4] 1 - in contact, 0 - not in contact
            contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
            contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0
            ### joints from abs traj
            foot_joint_rec_vel_from_abs_traj = (joint_pos_rec_from_abs_traj[:, 1:, self.foot_joint_index_list] -
                                                joint_pos_rec_from_abs_traj[:, 0:-1, self.foot_joint_index_list]) * self.fps  # [bs, clip_len-1, 4, 3]
            foot_joint_rec_vel_from_abs_traj = torch.norm(foot_joint_rec_vel_from_abs_traj, dim=-1)  # [bs, clip_len-1, 4]
            mask_skating_from_abs_traj = (foot_joint_rec_vel_from_abs_traj - self.foot_skating_vel_thres).gt(0)  # [bs, clip_len-1, 4]  True/False
            mask_skating_from_abs_traj = mask_skating_from_abs_traj * contact_lbl_rec[:, 0:-1]  # [bs, clip_len-1, 4]
            masked_foot_joint_rec_vel_from_abs_traj = foot_joint_rec_vel_from_abs_traj * mask_skating_from_abs_traj  # [bs, clip_len-1, 4]
            if mask_skating_from_abs_traj.sum() != 0:
                loss_foot_skating_from_abs_traj = masked_foot_joint_rec_vel_from_abs_traj.sum() / mask_skating_from_abs_traj.sum()
            else:
                loss_foot_skating_from_abs_traj = torch.tensor(0.0).to(self.device)

            ### joints from smpl
            foot_joint_rec_vel_from_smpl = (joint_pos_rec_from_smpl[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_smpl[:, 0:-1, self.foot_joint_index_list]) * self.fps
            foot_joint_rec_vel_from_smpl = torch.norm(foot_joint_rec_vel_from_smpl, dim=-1)
            mask_skating_from_smpl = (foot_joint_rec_vel_from_smpl - self.foot_skating_vel_thres).gt(0)
            mask_skating_from_smpl = mask_skating_from_smpl * contact_lbl_rec[:, 0:-1]
            masked_foot_joint_rec_vel_from_smpl = foot_joint_rec_vel_from_smpl * mask_skating_from_smpl
            if mask_skating_from_smpl.sum() != 0:
                loss_foot_skating_from_smpl = masked_foot_joint_rec_vel_from_smpl.sum() / mask_skating_from_smpl.sum()
            else:
                loss_foot_skating_from_smpl = torch.tensor(0.0).to(self.device)

            if mask_skating_from_abs_traj.sum() != 0 or mask_skating_from_smpl.sum() != 0:
                loss_foot_skating = loss_foot_skating_from_smpl + loss_foot_skating_from_abs_traj
                grad_skating = torch.autograd.grad([-loss_foot_skating], [x_t])[0]  # [bs, body_feat_dim, 1, T] same shape as x_t, body_feat_dim=294
                # print('loss_foot_skating_from_smpl: ', loss_foot_skating_from_smpl, 'loss_foot_skating_from_abs_traj',loss_foot_skating_from_abs_traj)
                grad_skating[:, 0:traj_feat_dim, :, :] = 0  # for traj repr part
                grad_skating[:, -4:, :, :] = 0  # for contact label
            else:
                grad_skating = torch.tensor(0.0).to(self.device)
            x_t.detach()

        return grad_skating


    def guide_2d_projection_with_smpl(self, batch, out, denoise_t, compute_grad='x_t'):
        # model_output: [bs, pose_feat_dim, 1, T]
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, body_feat_dim, 1, T]

            ########## obtain global joint coordinate
            full_repr_rec = x_t[:, :, 0].permute(0, 2, 1)  # [bs, T, body_feat_dim]
            full_repr_rec = full_repr_rec * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)

            # reconstruct joint positions
            traj_feat_dim = self.dataset.traj_feat_dim
            cur_total_dim = 0
            repr_dict_rec = {}
            for repr_name in REPR_LIST:
                repr_dict_rec[repr_name] = full_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
                cur_total_dim += REPR_DIM_DICT[repr_name]
            # joint_pos_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj', smplx_model=self.smplx_model)  # [bs, clip_len, 22, 3]
            # joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=self.smplx_model, return_full_joints=True)
            joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=self.smplx_model)

            ######### obtain joint coordinate in scene coord
            batch_size, clip_len = joint_pos_rec_from_smpl.shape[0], joint_pos_rec_from_smpl.shape[1]
            trans_cano2scene = torch.linalg.inv(batch['transf_matrix'])  # [bs, 4, 4]
            R, T = trans_cano2scene[:, 0:3, 0:3], trans_cano2scene[:, 0:3, -1]
            R = R.unsqueeze(1).repeat(1, clip_len, 1, 1).reshape(-1, 3, 3)
            T = T.unsqueeze(1).unsqueeze(1).repeat(1, clip_len, self.dataset.joints_num, 1).reshape(-1, self.dataset.joints_num, 3)
            temp = joint_pos_rec_from_smpl.reshape(batch_size*clip_len, -1, 3)
            joints_rec_scene_coord_from_smpl = torch.matmul(R, temp.permute(0, 2, 1)).permute(0, 2, 1) + T
            joints_rec_scene_coord_from_smpl = joints_rec_scene_coord_from_smpl.reshape(batch_size, clip_len, self.dataset.joints_num, 3)

            ###### trans to camera coord
            joints_rec_scene_coord_from_smpl = joints_rec_scene_coord_from_smpl.reshape(batch_size*clip_len, -1, 3)
            joints_rec_cam_coord = torch.matmul(torch.linalg.inv(self.dataset.cam_R), (joints_rec_scene_coord_from_smpl - self.dataset.cam_t).permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size*clip_len, 22, 3]
            ###### project to 2d
            focal_length = batch['focal_length'].unsqueeze(1).repeat(1, clip_len, 1).reshape(-1, 2)  # [batch_size*clip_len, 2]
            camera_center = batch['camera_center'].unsqueeze(1).repeat(1, clip_len, 1).reshape(-1, 2)
            joints_rec_2d = perspective_projection(points=joints_rec_cam_coord,
                                                   focal_length=focal_length,
                                                   camera_center=camera_center)  # [batch_size*clip_len, 22, 2]
            joints_rec_2d = joints_rec_2d.reshape(batch_size, clip_len, -1, 2)  # [batch_size, clip_len, 22, 2]
            loss_joints_2d = self.l1_loss(joints_rec_2d, batch['keypoints_2d'][:, :clip_len, :, 0:2])  # [batch_size, clip_len, 22, 2]
            loss_joints_2d = loss_joints_2d * batch['keypoints_2d'][:, :clip_len, :, [-1]]
            # currently select those joints in simplx main body 22 joints for 2d loss
            # todo: potentially we can add more joints for better alignment with image observations
            loss_joints_2d = loss_joints_2d[:, :, [16, 18, 20, 17, 19, 21, 4, 5, 7, 8]]
            loss_joints_2d = loss_joints_2d.mean()

            # print('loss_joints_2d: ', loss_joints_2d)
            grad_joints_2d = torch.autograd.grad([-loss_joints_2d], [x_t])[0]  # [bs, 263, 1, nframes] same shape as x_t
            grad_joints_2d[:, 0:traj_feat_dim, :, :] = 0
            grad_joints_2d[:, -4:, :, :] = 0
            x_t.detach()

        return grad_joints_2d






