"""
PoseNet test on AMASS, with ground truth trajectory given
"""
import argparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.dataloader_amass import DataloaderAMASS
from data_loaders.motion_representation import *

from model.posenet import PoseNet
from diffusion import gaussian_diffusion_posenet
from diffusion.respace import SpacedDiffusionPoseNet
from utils.model_util import create_gaussian_diffusion
from utils.vis_util import *
import smplx



group = argparse.ArgumentParser(description='RoHM code')
group.add_argument("--device", default=0, type=int, help="Device id to use.")
group.add_argument("--seed", default=0, type=int, help="For fixing random seed.")

######################## diffusion setups
group.add_argument("--diffusion_steps", default=1000, type=int, help='diffusion time steps')
group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
group.add_argument("--timestep_respacing_eval", default='', type=str)  # if use ddim, set to 'ddimN', where N denotes ddim sampling steps
group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'], help="Use smaller sigma values.")

######################## path to AMASS and body model
group.add_argument('--body_model_path', type=str, default='body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset_root', type=str, default='/mnt/hdd/diffusion_mocap_datasets/AMASS_smplx_preprocessed', help='path to datas')

####################### model setups
group.add_argument('--task', default='pose', type=str, choices=['traj', 'pose'])
group.add_argument("--clip_len", default=145, type=int, help="sequence length for each clip")
group.add_argument('--model_path', type=str, default='checkpoints/posenet_checkpoint/model000200000.pt', help='')

######################## input noise scaling setups
group.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'], help='if add nosie to input conditions')
group.add_argument("--noise_std_smplx_global_rot", default=3, type=float, help="noise ratio for smplx global orientation (unit: degree)")
group.add_argument("--noise_std_smplx_body_rot", default=2, type=float, help="noise ratio for smplx body pose (unit: degree)")
group.add_argument("--noise_std_smplx_trans", default=0.01, type=float, help="noise ratio for smplx global translation (unit: m)")
group.add_argument("--noise_std_smplx_betas", default=0.2, type=float, help="noise ratio for smplx shape param")

####################### test setups
group.add_argument("--batch_size", default=32, type=int, help="Batch size during test.")
group.add_argument('--cond_fn_with_grad', default='False', type=lambda x: x.lower() in ['true', '1'], help='use test-time guidance or not')
group.add_argument("--mask_scheme", default='lower', type=str, choices=['lower', 'upper', 'full'], help='occlusion setup for test')
group.add_argument('--visualize', default='True', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--vis_interval", default=50, type=int, help="visualize every N clips")
group.add_argument('--save_results', default='False', type=lambda x: x.lower() in ['true', '1'], help='save test results')


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
                                   repr_abs_only=False,
                                   input_noise=args.input_noise,
                                   noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                   noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                   noise_std_smplx_trans=args.noise_std_smplx_trans,
                                   noise_std_smplx_betas=args.noise_std_smplx_betas,
                                   task=args.task,
                                   clip_len=args.clip_len,
                                   logdir=log_dir,
                                   device=dist_util.dev())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    print("creating model and diffusion...")
    model = PoseNet(dataset=test_dataset, body_feat_dim=test_dataset.body_feat_dim,
                    latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                    body_model_path=args.body_model_path,
                    device=dist_util.dev(),
                    traj_feat_dim=test_dataset.traj_feat_dim,
                    ).to(dist_util.dev())

    print('[INFO] loaded model path:', args.model_path)
    weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
    model.eval()

    diffusion_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion_posenet,
                                               return_class=SpacedDiffusionPoseNet,
                                               num_diffusion_timesteps=args.diffusion_steps,
                                               timestep_respacing=args.timestep_respacing_eval,
                                               device=dist_util.dev())

    smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())

    ################## visualization
    if args.visualize:
        print('Visualizing...')
        print('[left - reconstruction]: [blue] visible parts / [yellow] occluded parts')
        print('[middle - noisy/occluded input]')
        print('[right - ground truth]: [red]')
        print('[foot contact label]: [red] not in contact with floor / [green] in contact with floor')
        import open3d as o3d
        from utils.other_utils import LIMBS_BODY_SMPL
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        cam_trans = np.array([[0, 0, -1, 5],
                              [-1, 0, 0, 2],
                              [0, -1, 0, 2],
                              [0, 0, 0, 1]])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame)
        from utils.other_utils import update_cam

    ############# data to save
    rec_ric_data_clean_list = []
    rec_ric_data_noisy_list = []
    rec_ric_data_rec_list_from_abs_traj = []
    rec_ric_data_rec_list_from_smpl = []
    motion_repr_clean_list = []
    motion_repr_noisy_list = []
    motion_repr_rec_list = []

    for test_step, test_batch in tqdm(enumerate(test_dataloader)):
        for key in test_batch.keys():
            test_batch[key] = test_batch[key].to(dist_util.dev())
        if not args.input_noise:
            test_batch['cond'] = test_batch['motion_repr_clean'].clone()  # [bs, clip_len, body_feat_dim]
        else:
            test_batch['cond'] = test_batch['motion_repr_noisy'].clone()
        bs, clip_len = test_batch['motion_repr_clean'].shape[0], test_batch['motion_repr_clean'].shape[1]

        ######################## mask out lower body part
        if args.mask_scheme == 'lower':
            mask_joint_id = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])
            for k in range(3):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + mask_joint_id * 3 + k] = 0.
            for k in range(3):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + 22 * 3 + mask_joint_id * 3 + k] = 0.
            for k in range(6):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id - 1) * 6 + k] = 0.
            test_batch['cond'][:, :, -4:] = 0.

        ######################## mask out upper body part
        if args.mask_scheme == 'upper':
            mask_joint_id = np.asarray([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            for k in range(3):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + mask_joint_id * 3 + k] = 0.
            for k in range(3):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + 22 * 3 + mask_joint_id * 3 + k] = 0.
            for k in range(6):
                test_batch['cond'][:, :, test_dataset.traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id - 1) * 6 + k] = 0.
            test_batch['cond'][:, :, -4:] = 0.

        ######################## mask out full body for certain ratio of frames
        if args.mask_scheme == 'full':
            start = torch.FloatTensor(bs).uniform_(0, clip_len - 1).long()  # [bs]
            mask_len = 30
            end = start + mask_len
            end[end > clip_len] = clip_len
            test_batch['cond'][:, :, -4:] = 0.
            for idx in range(bs):
                test_batch['cond'][idx, start[idx]:end[idx], 22:] = 0

        ######################## test forward
        test_batch['motion_repr_clean'] = torch.permute(test_batch['motion_repr_clean'], (0, 2, 1)).unsqueeze(-2)  # [bs, body_feat_dim, 1, clip_len]
        test_batch['cond'] = torch.permute(test_batch['cond'], (0, 2, 1)).unsqueeze(-2)
        shape = list(test_batch['motion_repr_clean'].shape)
        eval_losses, val_output = diffusion_eval.eval_losses(model=model, batch=test_batch,
                                                             shape=shape, progress=False,
                                                             clip_denoised=False,
                                                             timestep_respacing=args.timestep_respacing_eval,
                                                             cond_fn_with_grad=args.cond_fn_with_grad,
                                                             smplx_model=smplx_neutral)

        motion_repr_clean = test_batch['motion_repr_clean'][:, :, 0].permute(0, 2, 1).detach().cpu().numpy()  # [bs, clip_len, body_feat_dim]
        motion_repr_rec = val_output[:, :, 0].permute(0, 2, 1).detach().cpu().numpy()  # [bs, clip_len, body_feat_dim]
        if args.input_noise:
            motion_repr_noisy = test_batch['motion_repr_noisy'].detach().cpu().numpy()

        motion_repr_clean = motion_repr_clean * test_dataset.Std + test_dataset.Mean
        motion_repr_rec = motion_repr_rec * test_dataset.Std + test_dataset.Mean
        if args.input_noise:
            motion_repr_noisy = motion_repr_noisy * test_dataset.Std + test_dataset.Mean

        ######################## get joint positions
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

        ###### noisy input motion
        if args.input_noise:
            cur_total_dim = 0
            repr_dict_noisy = {}
            for repr_name in REPR_LIST:
                repr_dict_noisy[repr_name] = motion_repr_noisy[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
                repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
                cur_total_dim += REPR_DIM_DICT[repr_name]
            rec_ric_data_noisy, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            rec_ric_data_noisy = rec_ric_data_noisy.detach().cpu().numpy()

        ######################## save data
        if args.save_results:
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
            model_name = args.model_path.split('/')[-1][0:-3]
            pkl_path = os.path.join(log_dir, 'test_posenet_{}_guidance_{}.pkl'.format(model_name, args.cond_fn_with_grad))
            with open(pkl_path, 'wb') as result_file:
                pickle.dump(save_data, result_file, protocol=2)
            print('current data saved.')

        ################ get/calculate contact lbls
        contact_lbl_rec = motion_repr_rec[:, :, -4:]  # np, [bs, clip_len, 4]
        contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
        contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0
        contact_lbl_clean = motion_repr_clean[:, :, -4:]
        contact_lbl_clean[contact_lbl_clean > 0.5] = 1.0
        contact_lbl_clean[contact_lbl_clean <= 0.5] = 0.0

        ############################################## visualization
        if args.visualize:
            for bs in range(0, len(motion_repr_clean), 1):
                if bs % args.vis_interval == 0:
                    for t in range(len(rec_ric_data_rec_from_abs_traj[bs])):
                        ############################################# body skeletons
                        if args.mask_scheme == 'lower' or args.mask_scheme == 'upper':
                            cur_mask_joint_id = mask_joint_id.tolist()
                        else:
                            cur_mask_joint_id = None
                        skeleton_gt_list = vis_skeleton(joints=rec_ric_data_clean[bs, t], limbs=LIMBS_BODY_SMPL,
                                                        add_trans=np.array([0, 2.0, 2.5]))
                        skeleton_rec_list = vis_skeleton(joints=rec_ric_data_rec_from_smpl[bs, t],
                                                         limbs=LIMBS_BODY_SMPL, add_trans=np.array([0, 0.0, 2.5]),
                                                         mask_scheme=args.mask_scheme,
                                                         cur_mask_joint_id=cur_mask_joint_id)
                        if args.input_noise:
                            skeleton_noisy_list = vis_skeleton(joints=rec_ric_data_noisy[bs, t],
                                                               limbs=LIMBS_BODY_SMPL, add_trans=np.array([0, 1.0, 2.5]),
                                                               mask_scheme=args.mask_scheme,
                                                               cur_mask_joint_id=cur_mask_joint_id)

                        ############################################# foot contact labels
                        foot_sphere_clean_list = vis_foot_contact(joints=rec_ric_data_clean[bs, t],
                                                                  contact_lbl=contact_lbl_clean[bs, t],
                                                                  add_trans=np.array([0, 2.0, 0.0]))
                        foot_sphere_rec_list = vis_foot_contact(joints=rec_ric_data_rec_from_smpl[bs, t],
                                                                contact_lbl=contact_lbl_rec[bs, t])

                        ################# body mesh
                        body_mesh_clean = o3d.geometry.TriangleMesh()
                        body_mesh_clean.vertices = o3d.utility.Vector3dVector(smpl_verts_clean[bs, t].detach().cpu().numpy())
                        body_mesh_clean.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                        body_mesh_clean.compute_vertex_normals()
                        body_mesh_clean.paint_uniform_color(COLOR_GT_O3D)
                        transformation = np.identity(4)
                        transformation[1, 3] = 2.0
                        body_mesh_clean.transform(transformation)

                        body_mesh_noisy = o3d.geometry.TriangleMesh()
                        body_mesh_noisy.vertices = o3d.utility.Vector3dVector(smpl_verts_noisy[bs, t].detach().cpu().numpy())
                        body_mesh_noisy.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                        body_mesh_noisy.compute_vertex_normals()
                        body_mesh_noisy.paint_uniform_color(COLOR_VIS_O3D)
                        transformation = np.identity(4)
                        transformation[1, 3] = 1.0
                        body_mesh_noisy.transform(transformation)

                        body_mesh_rec = o3d.geometry.TriangleMesh()
                        body_mesh_rec.vertices = o3d.utility.Vector3dVector(smpl_verts_rec[bs, t].detach().cpu().numpy())
                        body_mesh_rec.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                        body_mesh_rec.compute_vertex_normals()
                        body_mesh_rec.paint_uniform_color(COLOR_VIS_O3D)

                        ################# visualize animation
                        vis.add_geometry(body_mesh_clean)
                        vis.add_geometry(body_mesh_noisy)
                        vis.add_geometry(body_mesh_rec)
                        for sphere in foot_sphere_clean_list:
                            vis.add_geometry(sphere)
                        for sphere in foot_sphere_rec_list:
                            vis.add_geometry(sphere)
                        for arrow in skeleton_gt_list:
                            vis.add_geometry(arrow)
                        for arrow in skeleton_rec_list:
                            vis.add_geometry(arrow)
                        if args.input_noise:
                            for arrow in skeleton_noisy_list:
                                vis.add_geometry(arrow)

                        ctr = vis.get_view_control()
                        cam_param = ctr.convert_to_pinhole_camera_parameters()
                        cam_param = update_cam(cam_param, cam_trans)
                        ctr.convert_from_pinhole_camera_parameters(cam_param)
                        vis.poll_events()
                        vis.update_renderer()
                        # time.sleep(0.03)

                        for sphere in foot_sphere_clean_list:
                            vis.remove_geometry(sphere)
                        for sphere in foot_sphere_rec_list:
                            vis.remove_geometry(sphere)
                        for arrow in skeleton_gt_list:
                            vis.remove_geometry(arrow)
                        for arrow in skeleton_rec_list:
                            vis.remove_geometry(arrow)
                        if args.input_noise:
                            for arrow in skeleton_noisy_list:
                                vis.remove_geometry(arrow)
                        vis.remove_geometry(body_mesh_clean)
                        vis.remove_geometry(body_mesh_noisy)
                        vis.remove_geometry(body_mesh_rec)

    print('test finished.')

if __name__ == "__main__":
    main(args)
    print('test finished.')
