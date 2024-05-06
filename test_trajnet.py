import argparse
from tqdm import tqdm
import smplx
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.dataloader_amass import DataloaderAMASS
from data_loaders.motion_representation import *

from model.trajnet import TrajNet
from diffusion import gaussian_diffusion_trajnet
from diffusion.respace import SpacedDiffusionTrajNet
from utils.model_util import create_gaussian_diffusion
from utils.other_utils import *



group = argparse.ArgumentParser(description='RoHM code')
group.add_argument("--device", default=0, type=int, help="Device id to use.")
group.add_argument("--seed", default=0, type=int, help="For fixing random seed.")

######################## diffusion setups
group.add_argument("--diffusion_steps", default=100, type=int, help='diffusion time steps')
group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
group.add_argument("--timestep_respacing_eval", default='', type=str)  # if use ddim, set to 'ddimN', where N denotes ddim sampling steps
group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'], help="Use smaller sigma values.")

######################## path to AMASS and body model
group.add_argument('--body_model_path', type=str, default='body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset_root', type=str, default='/mnt/hdd/diffusion_mocap_datasets/AMASS_smplx_preprocessed', help='path to datas')

######################## model setups
group.add_argument('--task', default='traj', type=str, choices=['traj', 'pose'])
group.add_argument("--clip_len", default=145, type=int, help="sequence length for each clip")
group.add_argument('--repr_abs_only', default='True', type=lambda x: x.lower() in ['true', '1'], help='if True, only include absolute trajectory repr')
group.add_argument("--trajcontrol", default=False, type=bool, help='if True, finetune trajnet with TrajControl')
group.add_argument('--model_path', type=str, default='checkpoints/trajnet_checkpoint/model000450000.pt', help='')

######################## input noise scaling setups
group.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'], help='if add nosie to input conditions')
group.add_argument("--noise_std_smplx_global_rot", default=1, type=float, help="noise ratio for smplx global orientation (unit: degree)")
group.add_argument("--noise_std_smplx_body_rot", default=1, type=float, help="noise ratio for smplx body pose (unit: degree)")
group.add_argument("--noise_std_smplx_trans", default=0.01, type=float, help="noise ratio for smplx global translation (unit: m)")
group.add_argument("--noise_std_smplx_betas", default=0.1, type=float, help="noise ratio for smplx shape param")

####################### test setups
group.add_argument("--batch_size", default=64, type=int, help="Batch size during test.")
group.add_argument('--infill_traj', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--max_infill_ratio", default=0.1, type=float, help="maximum occlusion ratio for traj infilling")
group.add_argument('--visualize', default='True', type=lambda x: x.lower() in ['true', '1'])


args = group.parse_args()
fixseed(args.seed)

def main(args):
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    amass_test_datasets = ['TCDHands', 'TotalCapture', 'SFU']
    # amass_test_datasets = ['SFU']

    log_dir = args.model_path.split('/')[0:-1]
    log_dir = '/'.join(log_dir)
    test_dataset = DataloaderAMASS(preprocessed_amass_root=args.dataset_root, split='test',
                                   amass_datasets=amass_test_datasets,
                                   body_model_path=args.body_model_path,
                                   repr_abs_only=args.repr_abs_only,
                                   input_noise=args.input_noise,
                                   noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                   noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                   noise_std_smplx_trans=args.noise_std_smplx_trans,
                                   noise_std_smplx_betas=args.noise_std_smplx_betas,
                                   task=args.task,
                                   clip_len=args.clip_len,
                                   logdir=log_dir,
                                   device=dist_util.dev()
                                   )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    print("creating model and diffusion...")
    model = TrajNet(time_dim=32, mid_dim=512,
                    cond_dim=test_dataset.traj_feat_dim, traj_feat_dim=test_dataset.traj_feat_dim,
                    trajcontrol=args.trajcontrol,
                    device=dist_util.dev(),
                    dataset=test_dataset,
                    repr_abs_only=args.repr_abs_only,
                    ).to(dist_util.dev())

    print('[INFO] loaded model path:', args.model_path)
    weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights)
    model.eval()

    diffusion_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_trajnet,
                                                return_class=SpacedDiffusionTrajNet,
                                                num_diffusion_timesteps=args.diffusion_steps,
                                                timestep_respacing=args.timestep_respacing_eval, device=dist_util.dev())

    smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())

    ################## visualization
    if args.visualize:
        import open3d as o3d
        from utils.other_utils import LIMBS_BODY_SMPL
        import time
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        cam_trans = np.array([[0, 0, -1, 5],
                              [-1, 0, 0, 2],
                              [0, -1, 0, 2],
                              [0, 0, 0, 1]])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame)
        from utils.other_utils import update_cam

    root_rot_err_rec_list = []
    root_x_err_rec_from_abs_traj_list, root_y_err_rec_from_abs_traj_list, root_z_err_rec_from_abs_traj_list = [], [], []
    root_x_err_rec_from_rel_traj_list, root_y_err_rec_from_rel_traj_list, root_z_err_rec_from_rel_traj_list = [], [], []
    root_x_err_rec_from_smpl_list, root_y_err_rec_from_smpl_list, root_z_err_rec_from_smpl_list = [], [], []

    root_pos_jitter_clean_list = []
    root_pos_jitter_noisy_list = []
    root_pos_jitter_rec_from_abs_traj_list = []
    root_pos_jitter_rec_from_rel_traj_list = []
    root_pos_jitter_rec_from_smpl_list = []

    fps = 30
    if args.visualize:
        print('Visualizing... [blue-prediction] [green-noisy input] [red-ground truth]')
    for test_step, test_batch in tqdm(enumerate(test_dataloader)):
        for key in test_batch.keys():
            test_batch[key] = test_batch[key].to(dist_util.dev())
        clip_len = test_batch['cond'].shape[1]
        batch_size = test_batch['cond'].shape[0]

        #################################### add mask
        if args.infill_traj:
            max_mask_ratio = args.max_infill_ratio
            start = torch.FloatTensor(batch_size).uniform_(0, clip_len - 1).long()
            mask_len = (clip_len * torch.FloatTensor(batch_size).uniform_(0, 1) * max_mask_ratio).long()
            end = start + mask_len
            end[end > clip_len] = clip_len
            mask_traj = torch.ones(batch_size, clip_len).to(dist_util.dev())  # [bs, t]
            for bs in range(batch_size):
                mask_traj[bs, start[bs]:end[bs]] = 0  # 1-visible 0-invisible
            mask_traj = mask_traj.unsqueeze(-1).repeat(1, 1, test_dataset.traj_feat_dim)  # [bs, t, 4]
            test_batch['cond'][:, :, 0:test_dataset.traj_feat_dim] = test_batch['cond'][:, :, 0:test_dataset.traj_feat_dim] * mask_traj
        #################################### add mask

        traj_feat_dim = test_dataset.traj_feat_dim
        shape = list(test_batch['motion_repr_clean'][:, :, 0:traj_feat_dim].shape)
        eval_losses, val_output = diffusion_eval.eval_losses(model=model, batch=test_batch,
                                                             shape=shape, progress=False,
                                                             clip_denoised=False,
                                                             timestep_respacing=args.timestep_respacing_eval,
                                                             cond_fn_with_grad=False,
                                                             smplx_model=smplx_neutral)
        if not args.repr_abs_only:
            motion_repr_clean_root_rec = torch.cat([val_output, test_batch['motion_repr_clean'][:, :, traj_feat_dim:]], dim=-1)
            motion_repr_clean_root_noisy = test_batch['motion_repr_clean'].clone()
            motion_repr_clean_root_noisy[:, :, 0:traj_feat_dim] = test_batch['motion_repr_noisy'][:, :, 0:traj_feat_dim]
        else:
            motion_repr_clean_root_rec = test_batch['motion_repr_clean'].clone()
            motion_repr_clean_root_rec[..., 0] = val_output[..., 0]
            motion_repr_clean_root_rec[..., 2:4] = val_output[..., 1:3]
            motion_repr_clean_root_rec[..., 6] = val_output[..., 3]
            motion_repr_clean_root_rec[..., 7:13] = val_output[..., 4:10]
            motion_repr_clean_root_rec[..., 16:19] = val_output[..., 10:13]
            motion_repr_clean_root_noisy = test_batch['motion_repr_clean'].clone()
            motion_repr_clean_root_noisy[..., 0] = test_batch['motion_repr_noisy'][..., 0]
            motion_repr_clean_root_noisy[..., 2:4] = test_batch['motion_repr_noisy'][..., 2:4]
            motion_repr_clean_root_noisy[..., 6] = test_batch['motion_repr_noisy'][..., 6]
            motion_repr_clean_root_noisy[..., 7:13] = test_batch['motion_repr_noisy'][..., 7:13]
            motion_repr_clean_root_noisy[..., 16:19] = test_batch['motion_repr_noisy'][..., 16:19]
        motion_repr_clean = test_batch['motion_repr_clean']

        motion_repr_clean = (motion_repr_clean.detach().cpu().numpy()) * test_dataset.Std + test_dataset.Mean
        motion_repr_clean_root_noisy = (motion_repr_clean_root_noisy.detach().cpu().numpy()) * test_dataset.Std + test_dataset.Mean
        motion_repr_clean_root_rec = (motion_repr_clean_root_rec.detach().cpu().numpy()) * test_dataset.Std + test_dataset.Mean

        ############# get joint locations
        ###### clean motion
        cur_total_dim = 0
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = motion_repr_clean[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_clean[repr_name] = torch.from_numpy(repr_dict_clean[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_clean, smpl_verts_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='smplx_params',
                                                                      smplx_model=smplx_neutral, return_verts=True)
        rec_ric_data_clean = rec_ric_data_clean.detach().cpu().numpy()

        ###### rec motion from abs traj / smpl params
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_rec[repr_name] = motion_repr_clean_root_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj',smplx_model=smplx_neutral)
        rec_ric_data_rec_from_abs_traj = rec_ric_data_rec_from_abs_traj.detach().cpu().numpy()
        rec_ric_data_rec_from_rel_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_rel_traj', smplx_model=smplx_neutral)
        rec_ric_data_rec_from_rel_traj = rec_ric_data_rec_from_rel_traj.detach().cpu().numpy()
        rec_ric_data_rec_from_smpl, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params',
                                                                            smplx_model=smplx_neutral,
                                                                            return_verts=True)
        rec_ric_data_rec_from_smpl = rec_ric_data_rec_from_smpl.detach().cpu().numpy()

        cur_total_dim = 0
        repr_dict_noisy = {}
        for repr_name in REPR_LIST:
            repr_dict_noisy[repr_name] = motion_repr_clean_root_noisy[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
            cur_total_dim += REPR_DIM_DICT[repr_name]
        rec_ric_data_noisy, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params',
                                                                      smplx_model=smplx_neutral, return_verts=True)
        rec_ric_data_noisy = rec_ric_data_noisy.detach().cpu().numpy()

        for bs in range(0, len(motion_repr_clean), 1):
            root_rot_clean = motion_repr_clean[bs, :, 0] * 2
            root_rot_noisy = motion_repr_clean_root_noisy[bs, :, 0] * 2
            root_rot_rec_from_abs_traj = motion_repr_clean_root_rec[bs, :, 0] * 2

            ###################### calculate error
            pelvis_traj_clean = rec_ric_data_clean[bs, :, 0]  # [clip_len, 3]
            pelvis_traj_noisy = rec_ric_data_noisy[bs, :, 0]
            pelvis_traj_rec_from_abs_traj = rec_ric_data_rec_from_abs_traj[bs, :, 0]
            pelvis_traj_rec_from_rel_traj = rec_ric_data_rec_from_rel_traj[bs, :, 0]
            pelvis_traj_rec_from_smpl = rec_ric_data_rec_from_smpl[bs, :, 0]

            root_rot_err_rec_list.append(np.abs(root_rot_rec_from_abs_traj - root_rot_clean))
            root_x_err_rec_from_abs_traj_list.append(np.abs(pelvis_traj_rec_from_abs_traj[:, 0] - pelvis_traj_clean[:, 0]))
            root_y_err_rec_from_abs_traj_list.append(np.abs(pelvis_traj_rec_from_abs_traj[:, 1] - pelvis_traj_clean[:, 1]))
            root_z_err_rec_from_abs_traj_list.append(np.abs(pelvis_traj_rec_from_abs_traj[:, 2] - pelvis_traj_clean[:, 2]))
            root_x_err_rec_from_rel_traj_list.append(np.abs(pelvis_traj_rec_from_rel_traj[:, 0] - pelvis_traj_clean[:, 0]))
            root_y_err_rec_from_rel_traj_list.append(np.abs(pelvis_traj_rec_from_rel_traj[:, 1] - pelvis_traj_clean[:, 1]))
            root_z_err_rec_from_rel_traj_list.append(np.abs(pelvis_traj_rec_from_rel_traj[:, 2] - pelvis_traj_clean[:, 2]))
            root_x_err_rec_from_smpl_list.append(np.abs(pelvis_traj_rec_from_smpl[:, 0] - pelvis_traj_clean[:, 0]))
            root_y_err_rec_from_smpl_list.append(np.abs(pelvis_traj_rec_from_smpl[:, 1] - pelvis_traj_clean[:, 1]))
            root_z_err_rec_from_smpl_list.append(np.abs(pelvis_traj_rec_from_smpl[:, 2] - pelvis_traj_clean[:, 2]))

            ############################### root position jitter, derivative of acceleration
            pelvis_jitter_clean = (pelvis_traj_clean[3:] - 3 * pelvis_traj_clean[2:-1] + 3 * pelvis_traj_clean[1:-2] - pelvis_traj_clean[:-3]) * (fps ** 3)  # [clip_len-3, 3]
            pelvis_jitter_clean = np.linalg.norm(pelvis_jitter_clean, axis=-1)  # # [clip_len-3]
            root_pos_jitter_clean_list.append(pelvis_jitter_clean)

            pelvis_jitter_noisy = (pelvis_traj_noisy[3:] - 3 * pelvis_traj_noisy[2:-1] + 3 * pelvis_traj_noisy[1:-2] - pelvis_traj_noisy[:-3]) * (fps ** 3)
            pelvis_jitter_noisy = np.linalg.norm(pelvis_jitter_noisy, axis=-1)
            root_pos_jitter_noisy_list.append(pelvis_jitter_noisy)

            pelvis_jitter_rec_from_abs_traj = (pelvis_traj_rec_from_abs_traj[3:] - 3 * pelvis_traj_rec_from_abs_traj[2:-1] + 3 * pelvis_traj_rec_from_abs_traj[1:-2] - pelvis_traj_rec_from_abs_traj[:-3]) * (fps ** 3)
            pelvis_jitter_rec_from_abs_traj = np.linalg.norm(pelvis_jitter_rec_from_abs_traj, axis=-1)
            root_pos_jitter_rec_from_abs_traj_list.append(pelvis_jitter_rec_from_abs_traj)

            pelvis_jitter_rec_from_rel_traj = (pelvis_traj_rec_from_rel_traj[3:] - 3 * pelvis_traj_rec_from_rel_traj[2:-1] + 3 * pelvis_traj_rec_from_rel_traj[1:-2] - pelvis_traj_rec_from_rel_traj[:-3]) * (fps ** 3)
            pelvis_jitter_rec_from_rel_traj = np.linalg.norm(pelvis_jitter_rec_from_rel_traj, axis=-1)
            root_pos_jitter_rec_from_rel_traj_list.append(pelvis_jitter_rec_from_rel_traj)

            pelvis_jitter_rec_from_smpl = (pelvis_traj_rec_from_smpl[3:] - 3 * pelvis_traj_rec_from_smpl[2:-1] + 3 * pelvis_traj_rec_from_smpl[1:-2] - pelvis_traj_rec_from_smpl[:-3]) * (fps ** 3)
            pelvis_jitter_rec_from_smpl = np.linalg.norm(pelvis_jitter_rec_from_smpl, axis=-1)
            root_pos_jitter_rec_from_smpl_list.append(pelvis_jitter_rec_from_smpl)

            ################################# visualization
            if args.visualize:
                if bs % 120 == 0:
                    if args.infill_traj:
                        mask = mask_traj[bs, :, 0].detach().cpu().numpy() == 1  # [T]
                    for t in range(rec_ric_data_rec_from_abs_traj.shape[1]):
                        color_gt = np.zeros([len(LIMBS_BODY_SMPL), 3])
                        color_gt[:, 0] = 1.0
                        skeleton_gt = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(rec_ric_data_clean[bs, t]),
                            lines=o3d.utility.Vector2iVector(LIMBS_BODY_SMPL))
                        skeleton_gt.colors = o3d.utility.Vector3dVector(color_gt)
                        transformation = np.identity(4)
                        transformation[1, 3] = 2.0
                        skeleton_gt.transform(transformation)

                        color_noisy = np.zeros([len(LIMBS_BODY_SMPL), 3])
                        color_noisy[:, 1] = 128/255
                        skeleton_noisy = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(rec_ric_data_noisy[bs, t]),
                            lines=o3d.utility.Vector2iVector(LIMBS_BODY_SMPL))
                        skeleton_noisy.colors = o3d.utility.Vector3dVector(color_noisy)
                        transformation = np.identity(4)
                        transformation[1, 3] = 1.0
                        skeleton_noisy.transform(transformation)

                        color_rec = np.zeros([len(LIMBS_BODY_SMPL), 3])
                        color_rec[:, 2] = 1.0
                        if args.infill_traj and mask[t] == 0:
                            color_rec = np.zeros([len(LIMBS_BODY_SMPL), 3])
                        skeleton_rec = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(rec_ric_data_rec_from_smpl[bs, t]),
                            lines=o3d.utility.Vector2iVector(LIMBS_BODY_SMPL))
                        skeleton_rec.colors = o3d.utility.Vector3dVector(color_rec)

                        skeleton_rec_cp = copy.deepcopy(skeleton_rec)
                        skeleton_gt_cp = copy.deepcopy(skeleton_gt)
                        transformation = np.identity(4)
                        transformation[1, 3] = -1.0
                        skeleton_rec_cp.transform(transformation)
                        transformation = np.identity(4)
                        transformation[1, 3] = -3.0
                        skeleton_gt_cp.transform(transformation)

                        vis.add_geometry(skeleton_gt)
                        vis.add_geometry(skeleton_noisy)
                        vis.add_geometry(skeleton_rec)
                        vis.add_geometry(skeleton_gt_cp)
                        vis.add_geometry(skeleton_rec_cp)

                        ctr = vis.get_view_control()
                        cam_param = ctr.convert_to_pinhole_camera_parameters()
                        cam_param = update_cam(cam_param, cam_trans)
                        ctr.convert_from_pinhole_camera_parameters(cam_param)

                        vis.poll_events()
                        vis.update_renderer()
                        time.sleep(0.05)

                        vis.remove_geometry(skeleton_gt)
                        vis.remove_geometry(skeleton_noisy)
                        vis.remove_geometry(skeleton_rec)
                        vis.remove_geometry(skeleton_gt_cp)
                        vis.remove_geometry(skeleton_rec_cp)



    import math
    print('[EVAL] {} clips in total.'.format(len(root_rot_err_rec_list)))
    root_rot_err_rec_list = np.concatenate(root_rot_err_rec_list, axis=0)
    root_x_err_rec_from_abs_traj_list = np.concatenate(root_x_err_rec_from_abs_traj_list, axis=0)
    root_y_err_rec_from_abs_traj_list = np.concatenate(root_y_err_rec_from_abs_traj_list, axis=0)
    root_z_err_rec_from_abs_traj_list = np.concatenate(root_z_err_rec_from_abs_traj_list, axis=0)
    print('[EVAL] root_rot_err_rec: {:0.3f}'.format(root_rot_err_rec_list.mean()), 'degree: {:0.2f}'.format(root_rot_err_rec_list.mean() * 180 / math.pi))
    print('[EVAL] root_x/y/z_err_rec_from_abs_traj (mm): {:0.2f} / {:0.2f} / {:0.2f}'
          .format(root_x_err_rec_from_abs_traj_list.mean() * 1000, root_y_err_rec_from_abs_traj_list.mean() * 1000, root_z_err_rec_from_abs_traj_list.mean() * 1000))

    root_x_err_rec_from_rel_traj_list = np.concatenate(root_x_err_rec_from_rel_traj_list, axis=0)
    root_y_err_rec_from_rel_traj_list = np.concatenate(root_y_err_rec_from_rel_traj_list, axis=0)
    root_z_err_rec_from_rel_traj_list = np.concatenate(root_z_err_rec_from_rel_traj_list, axis=0)
    print('[EVAL] root_x/y/z_err_rec_from_rel_traj (mm): {:0.2f} / {:0.2f} / {:0.2f}'
          .format(root_x_err_rec_from_rel_traj_list.mean() * 1000, root_y_err_rec_from_rel_traj_list.mean() * 1000, root_z_err_rec_from_rel_traj_list.mean() * 1000))

    root_x_err_rec_from_smpl_list = np.concatenate(root_x_err_rec_from_smpl_list, axis=0)
    root_y_err_rec_from_smpl_list = np.concatenate(root_y_err_rec_from_smpl_list, axis=0)
    root_z_err_rec_from_smpl_list = np.concatenate(root_z_err_rec_from_smpl_list, axis=0)
    print('[EVAL] root_x/y/z_err_rec_from_smpl (mm): {:0.2f} / {:0.2f} / {:0.2f}'
          .format(root_x_err_rec_from_smpl_list.mean() * 1000, root_y_err_rec_from_smpl_list.mean() * 1000, root_z_err_rec_from_smpl_list.mean() * 1000))


    root_pos_jitter_clean_list = np.concatenate(root_pos_jitter_clean_list, axis=0)
    root_pos_jitter_noisy_list = np.concatenate(root_pos_jitter_noisy_list, axis=0)
    root_pos_jitter_rec_from_abs_traj_list = np.concatenate(root_pos_jitter_rec_from_abs_traj_list, axis=0)
    root_pos_jitter_rec_from_rel_traj_list = np.concatenate(root_pos_jitter_rec_from_rel_traj_list, axis=0)
    root_pos_jitter_rec_from_smpl_list = np.concatenate(root_pos_jitter_rec_from_smpl_list, axis=0)
    root_pos_jitter_clean = sum(root_pos_jitter_clean_list) / len(root_pos_jitter_clean_list)
    root_pos_jitter_noisy = sum(root_pos_jitter_noisy_list) / len(root_pos_jitter_noisy_list)
    root_pos_jitter_rec_from_abs_traj = sum(root_pos_jitter_rec_from_abs_traj_list) / len(root_pos_jitter_rec_from_abs_traj_list)
    root_pos_jitter_rec_from_rel_traj = sum(root_pos_jitter_rec_from_rel_traj_list) / len(root_pos_jitter_rec_from_rel_traj_list)
    root_pos_jitter_rec_from_smpl = sum(root_pos_jitter_rec_from_smpl_list) / len(root_pos_jitter_rec_from_smpl_list)
    print('[EVAL] root_pos_jitter_clean / noisy / rec_from_abs_traj / rec_from_rel_traj / rec_from_smpl (m/s^3): {:0.2f} / {:0.2f} / {:0.2f} / {:0.2f} / {:0.2f}'
          .format(root_pos_jitter_clean, root_pos_jitter_noisy, root_pos_jitter_rec_from_abs_traj, root_pos_jitter_rec_from_rel_traj, root_pos_jitter_rec_from_smpl))




if __name__ == "__main__":
    main(args)
