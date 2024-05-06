import configargparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.dataloader_amass import DataloaderAMASS
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

######################## path to AMASS and body model
group.add_argument('--body_model_path', type=str, default='body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset_root', type=str, default='/mnt/hdd/diffusion_mocap_datasets/AMASS_smplx_preprocessed', help='path to datas')

####################### model setups
group.add_argument("--clip_len", default=145, type=int, help="sequence length for each clip")
group.add_argument('--repr_abs_only', default='True', type=lambda x: x.lower() in ['true', '1'], help='if True, only include absolute trajectory repr for TrajNet')
group.add_argument('--model_path_trajnet', type=str, default='../diffusion_mocap/runs_try/79530/model000450000.pt', help='')
group.add_argument('--model_path_trajnet_control', type=str, default='../diffusion_mocap/runs_try/65648/model000400000.pt', help='')
group.add_argument('--model_path_posenet', type=str, default='../diffusion_mocap/runs_try/54359/model000200000.pt', help='')

######################## input noise scaling setups
group.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--noise_std_smplx_global_rot", default=3, type=float, help=" ")
group.add_argument("--noise_std_smplx_body_rot", default=3, type=float, help=" ")
group.add_argument("--noise_std_smplx_trans", default=0.03, type=float, help=" ")
group.add_argument("--noise_std_smplx_betas", default=0.1, type=float, help=" ")
group.add_argument('--load_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--load_noise_level", default=3, type=int, help=" ")

####################### test setups
group.add_argument("--batch_size", default=32, type=int, help="Batch size during test.")
group.add_argument('--cond_fn_with_grad', default='True', type=lambda x: x.lower() in ['true', '1'], help='use test-time guidance or not')
group.add_argument('--infill_traj', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--traj_mask_ratio", default=0.1, type=float, help="occlusion ratio for traj infilling, when traj is occlude, we assume full body pose is also occluded")
group.add_argument("--mask_scheme", default='full', type=str, choices=['lower', 'upper', 'full'], help='occlusion scheme for poseNet')
group.add_argument('--save_root', type=str, default='test_results/results_amass_full', help='')

group.add_argument("--sample_iter", default=2, type=int, help="how many inference iterations during test, default is 2 for results in paper")
group.add_argument("--iter2_cond_noisy_traj", default='True', type=lambda x: x.lower() in ['true', '1'],
                   help='in inference iteration>1, if TrajNet conditions on noisy input instead of predicted traj from inderence iteration 1')
group.add_argument("--iter2_cond_noisy_pose", default='True', type=lambda x: x.lower() in ['true', '1'],
                   help='in inference iteration>1, if PoseNet conditions on noisy input instead of predicted pose from inderence iteration 1')
group.add_argument("--early_stop", default='False', type=lambda x: x.lower() in ['true', '1'],
                   help='if stop denoising earlier for PoseNet (for only 980 steps)')


args = group.parse_args()
fixseed(args.seed)

def main(args):
    dist_util.setup_dist(args.device)
    print("creating data loader...")
    amass_test_datasets = ['TCDHands', 'TotalCapture', 'SFU']
    # amass_test_datasets = ['SFU']

    ########### load pre-computed body noise
    if args.load_noise:
        noise_pkl_path = 'data/eval_noise_smplx/smplx_noise_level_{}.pkl'.format(args.load_noise_level)
        with open(noise_pkl_path, 'rb') as f:
            loaded_smplx_noise_dict = pickle.load(f)
    else:
        loaded_smplx_noise_dict = None

    log_dir_pose = args.model_path_posenet.split('/')[0:-1]
    log_dir_pose = '/'.join(log_dir_pose)
    test_pose_dataset = DataloaderAMASS(preprocessed_amass_root=args.dataset_root, split='test',
                                        amass_datasets=amass_test_datasets,
                                        body_model_path=args.body_model_path,
                                        input_noise=args.input_noise,
                                        noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                        noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                        noise_std_smplx_trans=args.noise_std_smplx_trans,
                                        noise_std_smplx_betas=args.noise_std_smplx_betas,
                                        load_noise=args.load_noise, loaded_smplx_noise_dict=loaded_smplx_noise_dict,
                                        task='pose',
                                        clip_len=args.clip_len,
                                        logdir=log_dir_pose,
                                        device=dist_util.dev())
    test_pose_dataloader = DataLoader(test_pose_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    test_pose_dataloader_iter = iter(test_pose_dataloader)

    ################## set up traj data loader
    log_dir_traj = args.model_path_trajnet.split('/')[0:-1]
    log_dir_traj = '/'.join(log_dir_traj)
    test_traj_dataset = DataloaderAMASS(preprocessed_amass_root=args.dataset_root, split='test',
                                        amass_datasets=amass_test_datasets,
                                        body_model_path=args.body_model_path,
                                        repr_abs_only=args.repr_abs_only,
                                        input_noise=args.input_noise,
                                        noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                        noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                        noise_std_smplx_trans=args.noise_std_smplx_trans,
                                        noise_std_smplx_betas=args.noise_std_smplx_betas,
                                        load_noise=args.load_noise, loaded_smplx_noise_dict=loaded_smplx_noise_dict,
                                        task='traj',
                                        clip_len=args.clip_len,
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
    rec_ric_data_clean_list = []
    rec_ric_data_noisy_list = []
    rec_ric_data_rec_list_from_abs_traj = []
    rec_ric_data_rec_list_from_smpl = []
    motion_repr_clean_list = []
    motion_repr_noisy_list = []
    motion_repr_rec_list = []

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
        for key in test_batch_pose.keys():
            test_batch_pose[key] = test_batch_pose[key].to(dist_util.dev())
        for key in test_batch_traj.keys():
            test_batch_traj[key] = test_batch_traj[key].to(dist_util.dev())

        if args.infill_traj:
            clip_len = test_batch_traj['cond'].shape[1]
            batch_size = test_batch_traj['cond'].shape[0]
            mask_traj = torch.ones(batch_size, clip_len).to(dist_util.dev())  # [bs, T]
            mask_len = int(args.traj_mask_ratio * 145)
            # default setup for tab.1 in the paper
            start = torch.ones([batch_size]).long() * 65
            end = start + mask_len
            for bs in range(batch_size):
                mask_traj[bs, start[bs]:end[bs]] = 0
            mask_traj = mask_traj.unsqueeze(-1).repeat(1, 1, test_traj_dataset.traj_feat_dim)  # [bs, T, traj_feat_dim]
            test_batch_traj['cond'][:, :, 0:test_traj_dataset.traj_feat_dim] = test_batch_traj['cond'][:, :, 0:test_traj_dataset.traj_feat_dim] * mask_traj

        for iter_idx in range(args.sample_iter):
            print('Inference iter {}...'.format(iter_idx))
            if args.iter2_cond_noisy_traj and args.infill_traj and iter_idx > 0:
                # for inference iter>0, TrajNet conditions on noisy visible input traj and predicted traj for occluded parts from last interence iteration
                traj_vis = test_batch_traj['cond'][:, :, 0:test_traj_dataset.traj_feat_dim] * mask_traj
                traj_occ = val_output_traj * (1-mask_traj)
                test_batch_traj['cond'][:, :, 0:test_traj_dataset.traj_feat_dim] = traj_vis + traj_occ

            ######################################## trajectory network forward  #########################################
            traj_feat_dim = test_traj_dataset.traj_feat_dim
            pose_feat_dim = test_traj_dataset.pose_feat_dim
            shape = list(test_batch_traj['motion_repr_clean'][:, :, 0:traj_feat_dim].shape)
            ################# for vanilla trajNet
            if iter_idx == 0: # val_output_traj: [bs, T-1, 13]
                _, val_output_traj = diffusion_trajnet_eval.eval_losses(model=model_trajnet, batch=test_batch_traj,
                                                                        shape=shape, progress=False,
                                                                        clip_denoised=False,
                                                                        timestep_respacing=args.timestep_respacing_eval,
                                                                        cond_fn_with_grad=args.cond_fn_with_grad,
                                                                        compute_loss=False,
                                                                        smplx_model=smplx_neutral)
                traj_noisy_full = test_batch_traj['motion_repr_noisy'][:, :, 0:22].detach().cpu().numpy()
            ################# for trajNet with trajControl
            else:
                # copy local pose from PoseNet to TrajControl condition
                test_batch_traj['control_cond'] = torch.zeros([shape[0], shape[1], pose_feat_dim]).to(dist_util.dev())
                test_batch_traj['control_cond'][:, 0:-1] = val_output_pose[:, :, 0].permute(0, 2, 1)[:, :, -pose_feat_dim:]
                test_batch_traj['control_cond'][:, -1] = test_batch_traj['control_cond'][:, -2].clone()
                _, val_output_traj = diffusion_trajnet_control_eval.eval_losses(model=model_trajnet_control,
                                                                                batch=test_batch_traj,
                                                                                shape=shape, progress=False,
                                                                                clip_denoised=False,
                                                                                timestep_respacing=args.timestep_respacing_eval,
                                                                                cond_fn_with_grad=args.cond_fn_with_grad,
                                                                                compute_loss=False,
                                                                                smplx_model=smplx_neutral)

            ################# motion_repr_clean_root_rec: full repr with reconstructed traj repr, pose part from gt (but unused)
            if not args.repr_abs_only:
                motion_repr_clean_root_rec = torch.cat([val_output_traj, test_batch_traj['motion_repr_clean'][:, :, traj_feat_dim:]], dim=-1)  # [bs, 144, 294]
            else:
                motion_repr_clean_root_rec = test_batch_traj['motion_repr_clean'].clone()
                motion_repr_clean_root_rec[..., 0] = val_output_traj[..., 0]
                motion_repr_clean_root_rec[..., 2:4] = val_output_traj[..., 1:3]
                motion_repr_clean_root_rec[..., 6] = val_output_traj[..., 3]
                motion_repr_clean_root_rec[..., 7:13] = val_output_traj[..., 4:10]
                motion_repr_clean_root_rec[..., 16:19] = val_output_traj[..., 10:13]
            if iter_idx == 0:
                test_batch_traj['motion_repr_noisy'] = motion_repr_clean_root_rec
            if iter_idx < args.sample_iter - 1 and not args.iter2_cond_noisy_traj:
                test_batch_traj['cond'] = val_output_traj
            motion_repr_clean_root_rec = motion_repr_clean_root_rec.detach().cpu().numpy()
            motion_repr_clean_root_rec = motion_repr_clean_root_rec * test_traj_dataset.Std + test_traj_dataset.Mean

            ################ reconstruct full traj repr (including both absolute and relative repr)
            cur_total_dim = 0
            repr_dict_clean_root_rec = {}
            for repr_name in REPR_LIST:
                repr_dict_clean_root_rec[repr_name] = motion_repr_clean_root_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
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
                test_batch_pose['motion_repr_clean'] = test_batch_pose['motion_repr_clean'][:, 0:-1]

            if not args.input_noise:
                if iter_idx == 0:
                    test_batch_pose['cond'] = test_batch_pose['motion_repr_clean'].clone()  # [bs, clip_len, body_feat_dim]
                else:
                    test_batch_pose['cond'] = test_batch_pose['motion_repr_clean'].clone()[:, :, 0].permute(0, 2, 1)  # [bs, clip_len, body_feat_dim]
            else:
                if args.iter2_cond_noisy_pose:
                    test_batch_pose['cond'] = test_batch_pose['motion_repr_noisy'].clone()
                else:
                    if iter_idx == 0:
                        test_batch_pose['cond'] = test_batch_pose['motion_repr_noisy'].clone()
                    else:
                        test_batch_pose['cond'] = val_output_pose[:, :, 0].permute(0, 2, 1)  # [bs, clip_len, body_feat_dim]
            #### replace condition traj with denoised output from traj network
            if not (args.mask_scheme == 'lower' and not args.input_noise):
                test_batch_pose['cond'][:, :, 0:22] = traj_rec_full
            bs, clip_len = test_batch_pose['motion_repr_clean'].shape[0], test_batch_pose['motion_repr_clean'].shape[1]

            ######### apply occlusion masks
            mask_iter_num = args.sample_iter if args.iter2_cond_noisy_pose else 1  # for iter inference>0, do not use occlusion mask if iter2_cond_noisy_pose=False
            if iter_idx < mask_iter_num:
                ######################## mask out lower body part
                if args.mask_scheme == 'lower':
                    mask_joint_id = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])
                    for k in range(3):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + mask_joint_id * 3 + k] = 0.
                    for k in range(3):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + 22 * 3 + mask_joint_id * 3 + k] = 0.
                    for k in range(6):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id - 1) * 6 + k] = 0.
                    test_batch_pose['cond'][:, :, -4:] = 0.
                ######################## mask out upper body part
                if args.mask_scheme == 'upper':
                    mask_joint_id = np.asarray([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20])
                    for k in range(3):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + mask_joint_id * 3 + k] = 0.
                    for k in range(3):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + 22 * 3 + mask_joint_id * 3 + k] = 0.
                    for k in range(6):
                        test_batch_pose['cond'][:, :, test_pose_dataset.traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id - 1) * 6 + k] = 0.
                    test_batch_pose['cond'][:, :, -4:] = 0.
                ######################## mask out full body pose (excluding traj) for some frames
                if args.mask_scheme == 'full':
                    if not args.infill_traj:
                        start = torch.FloatTensor(bs).uniform_(0, clip_len - 1).long()  # [bs]
                        mask_len = 30
                        end = start + mask_len
                        end[end > clip_len] = clip_len
                    test_batch_pose['cond'][:, :, -4:] = 0.
                    for idx in range(bs):
                        test_batch_pose['cond'][idx, start[idx]:end[idx], 22:] = 0

            test_batch_pose['cond'] = torch.permute(test_batch_pose['cond'], (0, 2, 1)).unsqueeze(-2)
            if iter_idx == 0:
                test_batch_pose['motion_repr_clean'] = torch.permute(test_batch_pose['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]

            shape = list(test_batch_pose['motion_repr_clean'].shape)
            print('PoseNet sampling...')
            _, val_output_pose = diffusion_posenet_eval.eval_losses(model=model_posenet, batch=test_batch_pose,
                                                                    shape=shape, progress=True,
                                                                    clip_denoised=False,
                                                                    timestep_respacing=args.timestep_respacing_eval,
                                                                    cond_fn_with_grad=args.cond_fn_with_grad,
                                                                    early_stop=args.early_stop,
                                                                    compute_loss=False,
                                                                    grad_type='amass',
                                                                    smplx_model=smplx_neutral)

        ####################################### get joint positions for input/output #######################################
        motion_repr_clean = test_batch_pose['motion_repr_clean'][:, :, 0].permute(0, 2, 1).detach().cpu().numpy()  # [bs, clip_len, body_feat_dim]
        motion_repr_rec = val_output_pose[:, :, 0].permute(0, 2, 1).detach().cpu().numpy()  # [bs, clip_len, body_feat_dim]
        if args.input_noise:
            motion_repr_noisy = test_batch_pose['motion_repr_noisy'].detach().cpu().numpy()
            motion_repr_noisy[:, :, 0:22] = traj_noisy_full[:, 0:-1, :]

        motion_repr_clean = motion_repr_clean * test_pose_dataset.Std + test_pose_dataset.Mean
        motion_repr_rec = motion_repr_rec * test_pose_dataset.Std + test_pose_dataset.Mean
        if args.input_noise:
            motion_repr_noisy = motion_repr_noisy * test_pose_dataset.Std + test_pose_dataset.Mean

        ############# get joint positions
        ###### clean motion
        cur_total_dim = 0
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = motion_repr_clean[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_clean[repr_name] = torch.from_numpy(repr_dict_clean[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_clean, smpl_verts_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
        rec_ric_data_clean = rec_ric_data_clean.detach().cpu().numpy()

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

        if args.input_noise:
            cur_total_dim = 0
            repr_dict_noisy = {}
            for repr_name in REPR_LIST:
                repr_dict_noisy[repr_name] = motion_repr_noisy[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
                repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
                cur_total_dim += REPR_DIM_DICT[repr_name]
            rec_ric_data_noisy, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            rec_ric_data_noisy = rec_ric_data_noisy.detach().cpu().numpy()

        ####################################### save data #######################################
        os.makedirs(args.save_root) if not os.path.exists(args.save_root) else None
        rec_ric_data_clean_list.append(rec_ric_data_clean)
        if args.input_noise:
            rec_ric_data_noisy_list.append(rec_ric_data_noisy)
        rec_ric_data_rec_list_from_abs_traj.append(rec_ric_data_rec_from_abs_traj)
        rec_ric_data_rec_list_from_smpl.append(rec_ric_data_rec_from_smpl)
        motion_repr_clean_list.append(motion_repr_clean)
        if args.input_noise:
            motion_repr_noisy_list.append(motion_repr_noisy)
        motion_repr_rec_list.append(motion_repr_rec)

        save_data = {}
        save_data['mask_scheme'] = args.mask_scheme
        save_data['repr_name_list'] = REPR_LIST
        save_data['repr_dim_dict'] = REPR_DIM_DICT
        save_data['rec_ric_data_clean_list'] = np.concatenate(rec_ric_data_clean_list, axis=0)
        if args.input_noise:
            save_data['rec_ric_data_noisy_list'] = np.concatenate(rec_ric_data_noisy_list, axis=0)
        save_data['rec_ric_data_rec_list_from_abs_traj'] = np.concatenate(rec_ric_data_rec_list_from_abs_traj, axis=0)
        save_data['rec_ric_data_rec_list_from_smpl'] = np.concatenate(rec_ric_data_rec_list_from_smpl, axis=0)
        save_data['motion_repr_clean_list'] = np.concatenate(motion_repr_clean_list, axis=0)
        if args.input_noise:
            save_data['motion_repr_noisy_list'] = np.concatenate(motion_repr_noisy_list, axis=0)
        save_data['motion_repr_rec_list'] = np.concatenate(motion_repr_rec_list, axis=0)
        save_dir = 'test_amass_full_grad_{}_mask_{}'.format(args.cond_fn_with_grad, args.mask_scheme)
        if args.input_noise and args.load_noise:
            save_dir += '_noise_{}'.format(args.load_noise_level)
        if args.infill_traj:
            save_dir += '_infill_traj_{}'.format(args.traj_mask_ratio)
        save_dir += '_iter_{}_iter2trajnoisy_{}_iter2posenoisy_{}_earlystop_{}_seed_{}.pkl'.\
            format(args.sample_iter, args.iter2_cond_noisy_traj, args.iter2_cond_noisy_pose, args.early_stop, args.seed)
        pkl_path = os.path.join(args.save_root, save_dir)
        with open(pkl_path, 'wb') as result_file:
            pickle.dump(save_data, result_file, protocol=2)
        print('current data saved.')

    print('test finished.')

if __name__ == "__main__":
    main(args)
