import configargparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.dataloader_video import DataloaderVideo
from data_loaders.motion_representation import *

from model.posenet import PoseNet
from diffusion import gaussian_diffusion_posenet
from model.trajnet import TrajNet
from diffusion import gaussian_diffusion_trajnet
from diffusion.respace import SpacedDiffusionPoseNet, SpacedDiffusionTrajNet
from utils.model_util import create_gaussian_diffusion
from utils.vis_util import *
import smplx


arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
cfg_parser = configargparse.YAMLConfigFileParser
description = 'RoHM code'
group = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='')
group.add_argument('--config', is_config_file=True, default='', help='config file path')
group.add_argument("--device", default=0, type=int, help="Device id to use.")
group.add_argument("--seed", default=0, type=int, help="For fixing random seed.")

######################## diffusion setups
group.add_argument("--diffusion_steps_posenet", default=1000, type=int, help='diffusion time steps')
group.add_argument("--diffusion_steps_trajnet", default=100, type=int, help='diffusion time steps')
group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
group.add_argument("--timestep_respacing_eval", default='', type=str)  # if use ddim, set to 'ddimN', where N denotes ddim sampling steps
group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'], help="Use smaller sigma values.")

######################## path to dataset and body model
group.add_argument('--body_model_path', type=str, default='body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset', type=str, default='egobody', choices=['prox', 'egobody'])
group.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to dataset')  # /mnt/hdd/PROX
group.add_argument('--init_root', type=str, default='data/init_motions/init_prox_rgb', help='path to initialized sequence')

####################### model setups
group.add_argument("--clip_len", default=145, type=int, help="sequence length for each clip")
group.add_argument('--repr_abs_only', default='True', type=lambda x: x.lower() in ['true', '1'], help='if True, only include absolute trajectory repr for TrajNet')
group.add_argument('--model_path_trajnet', type=str, default='../diffusion_mocap/runs_try/79530/model000450000.pt', help='')
group.add_argument('--model_path_trajnet_control', type=str, default='../diffusion_mocap/runs_try/65648/model000400000.pt', help='')
group.add_argument('--model_path_posenet', type=str, default='../diffusion_mocap/runs_try/54359/model000200000.pt', help='')

####################### test setups
group.add_argument("--batch_size", default=20, type=int, help="Batch size during test.")
group.add_argument('--cond_fn_with_grad', default='True', type=lambda x: x.lower() in ['true', '1'], help='use test-time guidance or not')
group.add_argument('--save_root', type=str, default='test_results/results_egobody', help='')

group.add_argument("--sample_iter", default=2, type=int, help="how many inference iterations during test, default is 2 for results in paper")
group.add_argument("--iter2_cond_noisy_traj", default='False', type=lambda x: x.lower() in ['true', '1'],
                   help='in inference iteration>1, if TrajNet conditions on noisy input instead of predicted traj from inderence iteration 1')
group.add_argument("--iter2_cond_noisy_pose", default='False', type=lambda x: x.lower() in ['true', '1'],
                   help='in inference iteration>1, if PoseNet conditions on noisy input instead of predicted pose from inderence iteration 1')
group.add_argument("--early_stop", default='True', type=lambda x: x.lower() in ['true', '1'],
                   help='if stop denoising earlier for PoseNet (for only 980 steps)')

group.add_argument("--window_size", default=2, type=int, help="sliding window.")
group.add_argument('--recording_name', type=str, default='recording_20211004_S12_S20_01', help='')
group.add_argument('--use_scene_floor_height', default='True', type=lambda x: x.lower() in ['true', '1'])

args = group.parse_args()
fixseed(args.seed)


def main(args):
    dist_util.setup_dist(args.device)
    print("creating data loader...")
    ################## set up joint MDM data loader
    log_dir_pose = args.model_path_posenet.split('/')[0:-1]
    log_dir_pose = '/'.join(log_dir_pose)
    test_pose_dataset = DataloaderVideo(dataset=args.dataset,
                                        init_root=args.init_root,
                                        base_dir=args.dataset_root,
                                        body_model_path=args.body_model_path,
                                        recording_name=args.recording_name,
                                        use_scene_floor_height=args.use_scene_floor_height,
                                        task='pose',
                                        clip_len=args.clip_len,
                                        overlap_len=args.window_size,
                                        logdir=log_dir_pose,
                                        device=dist_util.dev())
    test_pose_dataloader = DataLoader(test_pose_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    test_pose_dataloader_iter = iter(test_pose_dataloader)

    ################## set up traj data loader
    log_dir_traj = args.model_path_trajnet.split('/')[0:-1]
    log_dir_traj = '/'.join(log_dir_traj)
    test_traj_dataset = DataloaderVideo(dataset=args.dataset,
                                        init_root=args.init_root,
                                        base_dir=args.dataset_root,
                                        body_model_path=args.body_model_path,
                                        recording_name=args.recording_name,
                                        repr_abs_only=args.repr_abs_only,
                                        use_scene_floor_height=args.use_scene_floor_height,
                                        task='traj',
                                        clip_len=args.clip_len,
                                        overlap_len=args.window_size,
                                        logdir=log_dir_traj,
                                        device=dist_util.dev())
    test_traj_dataloader = DataLoader(test_traj_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    test_traj_dataloader_iter = iter(test_traj_dataloader)

    print("creating model and diffusion...")
    #################### set up PoseNet
    model_posenet = PoseNet(dataset=test_pose_dataset, body_feat_dim=test_pose_dataset.body_feat_dim,
                            latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                            body_model_path=args.body_model_path,
                            device=dist_util.dev(),
                            traj_feat_dim=test_pose_dataset.traj_feat_dim,
                            ).to(dist_util.dev())

    print('[INFO] loaded PoseNet checkpoint path:', args.model_path_posenet)
    weights = torch.load(args.model_path_posenet, map_location=lambda storage, loc: storage)
    model_posenet.load_state_dict(weights)
    model_posenet.eval()

    diffusion_posenet_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_posenet,
                                                       return_class=SpacedDiffusionPoseNet,
                                                       num_diffusion_timesteps=args.diffusion_steps_posenet,
                                                       timestep_respacing=args.timestep_respacing_eval,
                                                       device=dist_util.dev())

    #################### set up TrajNet
    model_trajnet = TrajNet(time_dim=32, mid_dim=512,
                            cond_dim=test_traj_dataset.traj_feat_dim,
                            traj_feat_dim=test_traj_dataset.traj_feat_dim,
                            trajcontrol=False,
                            device=dist_util.dev(),
                            dataset=test_traj_dataset,
                            repr_abs_only=args.repr_abs_only,
                            ).to(dist_util.dev())

    model_trajnet_control = TrajNet(time_dim=32, mid_dim=512,
                                    cond_dim=test_traj_dataset.traj_feat_dim,
                                    traj_feat_dim=test_traj_dataset.traj_feat_dim,
                                    trajcontrol=True,
                                    device=dist_util.dev(),
                                    dataset=test_traj_dataset,
                                    repr_abs_only=args.repr_abs_only,
                                    ).to(dist_util.dev())

    print('[INFO] loaded TrajNet checkpoint path:', args.model_path_trajnet)
    weights = torch.load(args.model_path_trajnet, map_location=lambda storage, loc: storage)
    model_trajnet.load_state_dict(weights)
    model_trajnet.eval()

    print('[INFO] loaded TrajNet TrajControl checkpoint path:', args.model_path_trajnet_control)
    weights = torch.load(args.model_path_trajnet_control, map_location=lambda storage, loc: storage)
    model_trajnet_control.load_state_dict(weights)
    model_trajnet_control.eval()

    diffusion_trajnet_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_trajnet,
                                                       return_class=SpacedDiffusionTrajNet,
                                                       num_diffusion_timesteps=args.diffusion_steps_trajnet,
                                                       timestep_respacing=args.timestep_respacing_eval,
                                                       device=dist_util.dev())
    diffusion_trajnet_control_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_trajnet,
                                                               return_class=SpacedDiffusionTrajNet,
                                                               num_diffusion_timesteps=args.diffusion_steps_trajnet,
                                                               timestep_respacing=args.timestep_respacing_eval,
                                                               device=dist_util.dev())

    smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())

    ############# data to save
    rec_ric_data_noisy_list = []
    rec_ric_data_rec_list_from_abs_traj = []
    rec_ric_data_rec_list_from_smpl = []
    joints_input_scene_coord_list = []
    joints_gt_scene_coord_list = []
    motion_repr_rec_list = []
    motion_repr_noisy_list = []
    mask_joint_vis_list = []
    trans_scene2cano_list = []


    for test_step in tqdm(range(len(test_pose_dataset) // args.batch_size + 1)):
        try:
            test_batch_pose = next(test_pose_dataloader_iter)
        except StopIteration:
            test_pose_dataloader_iter = iter(test_pose_dataloader)
            test_batch_pose = next(test_pose_dataloader_iter)
        try:
            test_batch_traj = next(test_traj_dataloader_iter)
        except StopIteration:
            test_traj_dataloader_iter = iter(test_traj_dataloader)
            test_batch_traj = next(test_traj_dataloader_iter)
        test_batch_traj['frame_name'] = np.asarray(test_batch_traj['frame_name']).T   # [bs, 145]
        test_batch_pose['frame_name'] = np.asarray(test_batch_pose['frame_name']).T  # [bs, 145]

        for key in test_batch_traj.keys():
            if key == 'cano_smplx_params_dict':
                for sub_key in test_batch_traj[key]:
                    test_batch_traj[key][sub_key] = test_batch_traj[key][sub_key].to(dist_util.dev())
            if key != 'frame_name' and key != 'cano_smplx_params_dict':
                test_batch_traj[key] = test_batch_traj[key].to(dist_util.dev())
        for key in test_batch_pose.keys():
            if key == 'cano_smplx_params_dict':
                for sub_key in test_batch_pose[key]:
                    test_batch_pose[key][sub_key] = test_batch_pose[key][sub_key].to(dist_util.dev())
            if key != 'frame_name' and key != 'cano_smplx_params_dict':
                test_batch_pose[key] = test_batch_pose[key].to(dist_util.dev())


        # sample_iter = args.sample_iter
        for iter_idx in range(args.sample_iter):
            print('Inference iter {}...'.format(iter_idx))
            ###############################################  trajectory network forward
            traj_feat_dim = test_traj_dataset.traj_feat_dim
            pose_feat_dim = test_traj_dataset.pose_feat_dim
            shape = list(test_batch_traj['motion_repr_noisy'][:, :, 0:traj_feat_dim].shape)
            ################# for vanilla trajNet
            if iter_idx == 0:  # val_output_traj: [bs, 144, 13]
                _, val_output_traj = diffusion_trajnet_eval.eval_losses(model=model_trajnet, batch=test_batch_traj,
                                                                        shape=shape, progress=False,
                                                                        clip_denoised=False,
                                                                        timestep_respacing=args.timestep_respacing_eval,
                                                                        compute_loss=False,
                                                                        cond_fn_with_grad=args.cond_fn_with_grad,
                                                                        smplx_model=smplx_neutral)
            ################# for trajNet with trajControl
            else:
                # copy local pose from PoseNet to TrajControl condition
                test_batch_traj['control_cond'] = torch.zeros([shape[0], shape[1], pose_feat_dim]).to(dist_util.dev())
                test_batch_traj['control_cond'][:, 0:-1] = val_output_joint[:, :, 0].permute(0, 2, 1)[:, :, -pose_feat_dim:]
                test_batch_traj['control_cond'][:, -1] = test_batch_traj['control_cond'][:, -2].clone()
                _, val_output_traj = diffusion_trajnet_control_eval.eval_losses(model=model_trajnet_control,
                                                                                batch=test_batch_traj,
                                                                                shape=shape, progress=False,
                                                                                clip_denoised=False,
                                                                                timestep_respacing=args.timestep_respacing_eval,
                                                                                cond_fn_with_grad=args.cond_fn_with_grad,
                                                                                compute_loss=False,
                                                                                smplx_model=smplx_neutral)

            ################# motion_repr_clean_root_rec: full repr with reconstructed traj repr, pose part from noisy input (but unused)
            if not args.repr_abs_only:
                motion_repr_input_root_rec = torch.cat([val_output_traj, test_batch_traj['motion_repr_noisy'][:, :, traj_feat_dim:]], dim=-1)
            else:
                motion_repr_input_root_rec = test_batch_traj['motion_repr_noisy'].clone()
                motion_repr_input_root_rec[..., 0] = val_output_traj[..., 0]
                motion_repr_input_root_rec[..., 2:4] = val_output_traj[..., 1:3]
                motion_repr_input_root_rec[..., 6] = val_output_traj[..., 3]
                motion_repr_input_root_rec[..., 7:13] = val_output_traj[..., 4:10]
                motion_repr_input_root_rec[..., 16:19] = val_output_traj[..., 10:13]
            if iter_idx == 0:
                test_batch_traj['motion_repr_noisy'] = motion_repr_input_root_rec
            if iter_idx < args.sample_iter - 1 and not args.iter2_cond_noisy_traj:
                test_batch_traj['cond'] = val_output_traj
            motion_repr_input_root_rec = motion_repr_input_root_rec.detach().cpu().numpy()
            motion_repr_input_root_rec = motion_repr_input_root_rec * test_traj_dataset.Std + test_traj_dataset.Mean

            ################ reconstruct full traj repr (including both absolute and relative repr)
            cur_total_dim = 0
            repr_dict_clean_root_rec = {}
            for repr_name in REPR_LIST:
                repr_dict_clean_root_rec[repr_name] = motion_repr_input_root_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
                repr_dict_clean_root_rec[repr_name] = torch.from_numpy(repr_dict_clean_root_rec[repr_name]).to(dist_util.dev())
                cur_total_dim += REPR_DIM_DICT[repr_name]
            rec_ric_data_rec_from_smpl, smpl_verts_rec = recover_from_repr_smpl(repr_dict_clean_root_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            rec_ric_data_rec_from_smpl = rec_ric_data_rec_from_smpl.detach().cpu().numpy()

            traj_rec_full = []
            for seq_i in range(len(rec_ric_data_rec_from_smpl)):
                global_orient_mat = rot6d_to_rotmat(repr_dict_clean_root_rec['smplx_rot_6d'][seq_i])  # [T, 3, 3]
                global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat)  # [T, 3]
                body_pose_mat = rot6d_to_rotmat(repr_dict_clean_root_rec['smplx_body_pose_6d'][seq_i].reshape(-1, 6))  # [T*21, 3, 3]
                body_pose_aa = rotation_matrix_to_angle_axis(body_pose_mat).reshape(-1, 21, 3)  # [T, 21, 3]
                smplx_params_dict = {'transl': repr_dict_clean_root_rec['smplx_trans'][seq_i].detach().cpu().numpy(),
                                     'global_orient': global_orient_aa.detach().cpu().numpy(),
                                     'body_pose': body_pose_aa.reshape(-1, 63).detach().cpu().numpy(),
                                     'betas': repr_dict_clean_root_rec['smplx_betas'][seq_i].detach().cpu().numpy(), }
                repr_dict = get_repr_smplx(positions=rec_ric_data_rec_from_smpl[seq_i], smplx_params_dict=smplx_params_dict,
                                           feet_vel_thre=5e-5)  # a dict of reprs
                new_motion_repr_clean_root_rec = np.concatenate([repr_dict[key] for key in REPR_LIST], axis=-1)
                new_motion_repr_clean_root_rec = (new_motion_repr_clean_root_rec - test_pose_dataset.Mean) / test_pose_dataset.Std
                traj_rec_full.append(new_motion_repr_clean_root_rec[:, 0:22])
            traj_rec_full = np.asarray(traj_rec_full)  # [bs, 143, 22]
            traj_rec_full = torch.tensor(traj_rec_full).to(dist_util.dev())

            ######################################### PoseNet forward  #####################################
            if iter_idx == 0:
                test_batch_pose['motion_repr_noisy'] = test_batch_pose['motion_repr_noisy'][:, 0:-1]  # T=144-->143
            if args.iter2_cond_noisy_pose:
                test_batch_pose['cond'] = test_batch_pose['motion_repr_noisy'].clone()
                if iter_idx > 0:
                    test_batch_pose['cond'] = test_batch_pose['cond'][:, :, 0].permute(0, 2, 1)
            else:
                if iter_idx == 0:
                    test_batch_pose['cond'] = test_batch_pose['motion_repr_noisy'].clone()
                else:
                    test_batch_pose['cond'] = val_output_joint[:, :, 0].permute(0, 2, 1)  # [bs, clip_len, body_feat_dim]
            #### replace condition traj with denoised output from traj network
            test_batch_pose['cond'][:, :, 0:22] = traj_rec_full

            ######### apply occlusion masks
            mask_iter_num = args.sample_iter if args.iter2_cond_noisy_pose else 1  # for iter inference>0, do not use occlusion mask if iter2_cond_noisy_pose=False
            if iter_idx < mask_iter_num:
                mask_joint_vis = test_batch_pose['mask_joint_vis'][:, 0:-2, :]  # [bs, T-2, 22]
                test_batch_pose['cond'] = test_batch_pose['cond'] * test_batch_pose['mask_vec_vis'][:, 0:-2, :]
                test_batch_pose['cond'][:, :, -4:] = 0.

            if iter_idx == 0:
                test_batch_pose['motion_repr_noisy'] = torch.permute(test_batch_pose['motion_repr_noisy'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
            test_batch_pose['cond'] = torch.permute(test_batch_pose['cond'], (0, 2, 1)).unsqueeze(-2)

            shape = list(test_batch_pose['motion_repr_noisy'].shape)
            _, val_output_joint = diffusion_posenet_eval.eval_losses(model=model_posenet, batch=test_batch_pose,
                                                                     shape=shape, progress=True,
                                                                     clip_denoised=False,
                                                                     timestep_respacing=args.timestep_respacing_eval,
                                                                     cond_fn_with_grad=args.cond_fn_with_grad,
                                                                     early_stop=args.early_stop,
                                                                     compute_loss=False,
                                                                     grad_type='prox',
                                                                     smplx_model=smplx_neutral)

        ####################################### get joint positions for input/output #######################################
        motion_repr_rec = val_output_joint[:, :, 0].permute(0, 2, 1).detach().cpu().numpy()  # [bs, clip_len, body_feat_dim]
        motion_repr_noisy = test_batch_pose['motion_repr_noisy'][:, :, 0].permute(0, 2, 1).detach().cpu().numpy()

        motion_repr_rec = motion_repr_rec * test_pose_dataset.Std + test_pose_dataset.Mean
        motion_repr_noisy = motion_repr_noisy * test_pose_dataset.Std + test_pose_dataset.Mean

        ############# get joint positions
        ###### noisy/occluded input motion
        cur_total_dim = 0
        repr_dict_noisy = {}
        for repr_name in REPR_LIST:
            repr_dict_noisy[repr_name] = motion_repr_noisy[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_noisy, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
        rec_ric_data_noisy = rec_ric_data_noisy.detach().cpu().numpy()

        ###### rec motion from abs traj / smpl params
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_rec[repr_name] = motion_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj', smplx_model=smplx_neutral)
        rec_ric_data_rec_from_abs_traj = rec_ric_data_rec_from_abs_traj.detach().cpu().numpy()
        rec_ric_data_rec_from_smpl, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
        rec_ric_data_rec_from_smpl = rec_ric_data_rec_from_smpl.detach().cpu().numpy()

        ############################################################## save data
        os.makedirs(args.save_root) if not os.path.exists(args.save_root) else None
        trans_scene2cano_list.append(test_batch_pose['transf_matrix'].detach().cpu().numpy())
        rec_ric_data_noisy_list.append(rec_ric_data_noisy)
        rec_ric_data_rec_list_from_abs_traj.append(rec_ric_data_rec_from_abs_traj)
        rec_ric_data_rec_list_from_smpl.append(rec_ric_data_rec_from_smpl)
        joints_input_scene_coord_list.append(test_batch_pose['noisy_joints_scene_coord'].detach().cpu().numpy())
        if args.dataset == 'egobody':
            joints_gt_scene_coord_list.append(test_batch_pose['gt_joints_scene_coord'].detach().cpu().numpy())
        motion_repr_rec_list.append(motion_repr_rec)
        motion_repr_noisy_list.append(motion_repr_noisy)
        mask_joint_vis_list.append(mask_joint_vis.detach().cpu().numpy())

        save_data = {}
        if args.dataset == 'egobody':
            save_data['gender_gt'] = test_pose_dataset.gender_gt
            save_data['joints_gt_scene_coord_list'] = np.concatenate(joints_gt_scene_coord_list, axis=0)
        save_data['repr_name_list'] = REPR_LIST
        save_data['repr_dim_dict'] = REPR_DIM_DICT
        save_data['frame_name_list'] = test_batch_pose['frame_name']
        save_data['trans_scene2cano_list'] = np.concatenate(trans_scene2cano_list, axis=0)
        save_data['rec_ric_data_noisy_list'] = np.concatenate(rec_ric_data_noisy_list, axis=0)
        save_data['rec_ric_data_rec_list_from_abs_traj'] = np.concatenate(rec_ric_data_rec_list_from_abs_traj, axis=0)
        save_data['rec_ric_data_rec_list_from_smpl'] = np.concatenate(rec_ric_data_rec_list_from_smpl, axis=0)
        save_data['joints_input_scene_coord_list'] = np.concatenate(joints_input_scene_coord_list, axis=0)
        save_data['motion_repr_noisy_list'] = np.concatenate(motion_repr_noisy_list, axis=0)
        save_data['motion_repr_rec_list'] = np.concatenate(motion_repr_rec_list, axis=0)
        save_data['mask_joint_vis_list'] = np.concatenate(mask_joint_vis_list, axis=0)  # [n_clip, 143, 22]
        save_data['recording_name'] = test_pose_dataset.recording_name
        ####### write data to disk
        save_dir = 'test_{}_grad_{}_iter_{}_iter2trajnoisy_{}_iter2posenoisy_{}_earlystop_{}_seed_{}'.\
            format(args.dataset, args.cond_fn_with_grad, args.sample_iter, args.iter2_cond_noisy_traj, args.iter2_cond_noisy_pose, args.early_stop, args.seed)
        save_dir = os.path.join(args.save_root, save_dir)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        pkl_path = os.path.join(save_dir, '{}.pkl'.format(test_pose_dataset.recording_name))
        with open(pkl_path, 'wb') as result_file:
            pickle.dump(save_data, result_file, protocol=2)
        print('current data saved.')

    print('test finished.')


if __name__ == "__main__":
    main(args)
