import pickle
from data_loaders.motion_representation import *
from utils import dist_util
from utils.vis_util import *
from utils.render_util import *
import smplx
import configargparse
from PIL import Image


arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
cfg_parser = configargparse.YAMLConfigFileParser
description = 'RoHM code'
group = configargparse.ArgParser(formatter_class=arg_formatter,
                                 config_file_parser_class=cfg_parser,
                                 description=description,
                                 prog='')
group.add_argument('--config', is_config_file=True, default='', help='config file path')
group.add_argument("--device", default=0, type=int, help="Device id to use.")
group.add_argument('--body_model_path', type=str, default='data/body_models/smplx_model', help='path to smplx model')
group.add_argument('--saved_data_path', type=str,
                   default='data/test_results_release/results_amass_full/test_amass_full_grad_True_mask_lower_noise_3_iter_2_iter2trajnoisy_True_iter2posenoisy_True_earlystop_False_seed_0.pkl',
                   help='path to saved test results')
group.add_argument("--mask_scheme", default='lower', type=str, choices=['lower', 'full'], help='occlusion setup for test, full denotes traj+body occluded together')
group.add_argument("--traj_mask_ratio", default=0.0, type=float, help="occlusion ratio for traj infilling, when traj is occlude, we assume full body pose is also occluded")

group.add_argument('--visualize', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--vis_interval", default=100, type=int, help="visualize every N clips")

group.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--render_interval", default=100, type=int, help="render every N clips")
group.add_argument("--render_save_path", default='render_imgs/render_amass/mask_lower_noise_3', type=str, help='path to save render images')



args = group.parse_args()
dist_util.setup_dist(args.device)
smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())


if __name__ == "__main__":
    with open(args.saved_data_path, 'rb') as f:
        saved_data = pickle.load(f)
    print(args.saved_data_path)

    if args.visualize and args.render:
        print('[ERROR] cannot visualize and render at the same time.')
        exit()

    ################################# read test results data
    repr_name_list = saved_data['repr_name_list']
    repr_dim_dict = saved_data['repr_dim_dict']
    rec_ric_data_clean_list = saved_data['rec_ric_data_clean_list']  # [n_seq, clip_len, 22, 3]
    input_noise = False
    if 'rec_ric_data_noisy_list' in saved_data.keys():
        rec_ric_data_noisy_list = saved_data['rec_ric_data_noisy_list']
        input_noise = True
    rec_ric_data_rec_list_from_abs_traj = saved_data['rec_ric_data_rec_list_from_abs_traj']
    rec_ric_data_rec_list_from_smpl = saved_data['rec_ric_data_rec_list_from_smpl']
    motion_repr_clean_list = saved_data['motion_repr_clean_list']
    if input_noise == True:
        motion_repr_noisy_list = saved_data['motion_repr_noisy_list']
    else:
        motion_repr_noisy_list = saved_data['motion_repr_clean_list']
    motion_repr_rec_list = saved_data['motion_repr_rec_list']
    n_seq = len(rec_ric_data_clean_list)
    clip_len = rec_ric_data_clean_list.shape[1]
    print('n_seq: ', n_seq)

    ################# mpjpe for all/visible/occluded joints
    joints_mpjpe_global = np.linalg.norm(rec_ric_data_clean_list - rec_ric_data_rec_list_from_smpl, axis=-1)  # [n_seq, clip_len, 22]
    print('mpjpe_global (mm): {:0.1f}'.format(np.mean(joints_mpjpe_global) * 1000))
    start, end = 0, 0
    if args.mask_scheme == 'lower':
        mask_joint_id = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])
        vis_joint_id = set(range(22)) - set(mask_joint_id)
        joints_mpjpe_global_vis = joints_mpjpe_global[:, :, list(vis_joint_id)]
        joints_mpjpe_global_invis = joints_mpjpe_global[:, :, mask_joint_id]
        print('mpjpe_global_vis / occ (mm): {:0.1f} / {:0.1f}'.format(np.mean(joints_mpjpe_global_vis) * 1000, np.mean(joints_mpjpe_global_invis) * 1000))
    elif args.mask_scheme == 'full':
        # default setup for tab.1 in the paper
        start = 65
        mask_len = int(args.traj_mask_ratio * 145)
        end = start + mask_len
        joints_mpjpe_global_vis = np.concatenate([joints_mpjpe_global[:, 0:start, ], joints_mpjpe_global[:, end:, ]], axis=1)
        joints_mpjpe_global_invis = joints_mpjpe_global[:, start:end, ]
        print('mpjpe_global_vis / occ (mm): {:0.1f} / {:0.1f}'.format(np.mean(joints_mpjpe_global_vis) * 1000, np.mean(joints_mpjpe_global_invis) * 1000))

    ################ calculate contact lbls acc
    contact_lbl_rec_list = motion_repr_rec_list[:, :, -4:]  # np, [n_seq, clip_len, 4]
    contact_lbl_rec_list[contact_lbl_rec_list > 0.5] = 1.0
    contact_lbl_rec_list[contact_lbl_rec_list <= 0.5] = 0.0
    contact_lbl_clean_list = motion_repr_clean_list[:, :, -4:]
    contact_lbl_acc = (contact_lbl_clean_list == contact_lbl_rec_list)
    print('contact_lbl_acc: {:0.2f}'.format(np.mean(contact_lbl_acc)))


    ############################### foot skating ratio
    thresh_height = 0.10
    thresh_vel = 0.10
    fps = 30
    foot_joint_index_list = [7, 10, 8, 11]  # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe

    min_height_gt = rec_ric_data_clean_list[:, :, :, 2].min(axis=-1).min(axis=-1)  # [n_seq]
    joints_foot_gt = rec_ric_data_clean_list[:, :, foot_joint_index_list, :]  # [n_seq, clip_len, 4, 3]
    joints_feet_horizon_vel_gt = np.linalg.norm(joints_foot_gt[:, 1:, :, [0, 1]] - joints_foot_gt[:, :-1, :, [0, 1]],
                                                axis=-1) * fps  # [n_seq, clip_len, 4]
    joints_feet_height_gt = joints_foot_gt[:, 0:-1, :, 2]  # [n_seq, clip_len, 4]
    joints_feet_height_gt = joints_feet_height_gt - np.tile(min_height_gt.reshape(len(min_height_gt), 1, 1),
                                                            (1, joints_feet_height_gt.shape[-2], len(foot_joint_index_list)))  # [n_seq, clip_len-1, 4]
    skating_gt_left = (joints_feet_horizon_vel_gt[:, :, 0] > thresh_vel) * (joints_feet_horizon_vel_gt[:, :, 1] > thresh_vel) * \
                      (joints_feet_height_gt[:, :, 0] < (thresh_height + 0.05)) * (joints_feet_height_gt[:, :, 1] < thresh_height)
    skating_gt_right = (joints_feet_horizon_vel_gt[:, :, 2] > thresh_vel) * (joints_feet_horizon_vel_gt[:, :, 3] > thresh_vel) * \
                       (joints_feet_height_gt[:, :, 2] < (thresh_height + 0.05)) * (joints_feet_height_gt[:, :, 3] < thresh_height)
    skating_gt = skating_gt_left * skating_gt_right
    skating_gt_ratio = np.mean(skating_gt)
    print('skating_gt_ratio: {:0.3f}'.format(skating_gt_ratio))

    joints_foot_rec = rec_ric_data_rec_list_from_smpl[:, :, foot_joint_index_list, :]  # [n_seq, clip_len, 2, 3]
    joints_feet_horizon_vel_rec = np.linalg.norm(joints_foot_rec[:, 1:, :, [0, 1]] - joints_foot_rec[:, :-1, :, [0, 1]],
                                                axis=-1) * fps  # [n_seq, clip_len, 2]
    joints_feet_height_rec = joints_foot_rec[:, 0:-1, :, 2]  # [n_seq, clip_len, 2]
    joints_feet_height_rec = joints_feet_height_rec - np.tile(min_height_gt.reshape(len(min_height_gt), 1, 1),
                                                            (1, joints_feet_height_rec.shape[-2], len(foot_joint_index_list)))  # todo: use min_height_rec or min_height_gt
    skating_rec_left = (joints_feet_horizon_vel_rec[:, :, 0] > thresh_vel) * (joints_feet_horizon_vel_rec[:, :, 1] > thresh_vel) * \
                      (joints_feet_height_rec[:, :, 0] < (thresh_height + 0.05)) * (joints_feet_height_rec[:, :, 1] < thresh_height)
    skating_rec_right = (joints_feet_horizon_vel_rec[:, :, 2] > thresh_vel) * (joints_feet_horizon_vel_rec[:, :, 3] > thresh_vel) * \
                       (joints_feet_height_rec[:, :, 2] < (thresh_height + 0.05)) * (joints_feet_height_rec[:, :, 3] < thresh_height)
    skating_rec = skating_rec_left * skating_rec_right  # [n_seq, t]
    skating_rec_ratio = np.mean(skating_rec)
    print('skating_rec_ratio: {:0.3f}'.format(skating_rec_ratio))

    ########################### acceleration metrics
    acc_rec = (rec_ric_data_rec_list_from_smpl[:, 2:] - 2 * rec_ric_data_rec_list_from_smpl[:, 1:-1] + rec_ric_data_rec_list_from_smpl[:, :-2]) * (fps ** 2)  # [n_clip, 141, 22, 3]
    acc_gt = (rec_ric_data_clean_list[:, 2:] - 2 * rec_ric_data_clean_list[:, 1:-1] + rec_ric_data_clean_list[:, :-2]) * (fps ** 2)  # [n_clip, 141, 22, 3]
    acc_error = np.linalg.norm(acc_rec - acc_gt, axis=-1).mean()
    print('accel_error (m/s^2): {:0.1f}'.format(acc_error))

    ########################### ground penetration metrics
    pene_dist = rec_ric_data_rec_list_from_smpl[:, :, [10, 11], -1] - np.tile(min_height_gt.reshape(n_seq, 1, 1), (1, clip_len, 2))  # [n_seq, clip_len, 2]
    pene_freq = pene_dist < -0.05  # [clip_len]
    pene_freq = pene_freq.mean()
    pene_dist[pene_dist >= 0] = 0
    pene_dist = pene_dist.mean()
    print('ground_pene_freq score (%): {:0.2f}'.format(pene_freq*100))
    print('ground_pene_dist score (mm): {:0.2f}'.format(pene_dist*1000))

    ################## visualization
    if args.visualize:
        import open3d as o3d
        from utils.other_utils import *
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        cam_trans = np.array([[0, 0, -1, 5],
                              [-1, 0, 0, 2],
                              [0, -1, 0, 2],
                              [0, 0, 0, 1]])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame)

        ground = o3d.geometry.TriangleMesh.create_box(width=10.0,
                                                        height=10.0,
                                                        depth=0.1)
        ground_trans = np.array([[1, 0, 0, -5],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, -0.1],
                                   [0, 0, 0, 1]])
        ground.transform(ground_trans)
        ground.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(ground)

        ############################################## visualization
        print('Visualizing...')
        print('[left - reconstruction]: [blue] visible parts / [yellow] occluded parts')
        print('[middle - noisy/occluded input]')
        print('[right - ground truth]: [red]')
        print('[foot contact label]: [red] not in contact with floor / [green] in contact with floor')
        for bs in range(0, n_seq, 1):
            if bs % args.vis_interval == 0:
                cur_total_dim = 0
                repr_dict_clean = {}
                repr_dict_rec = {}
                repr_dict_noisy = {}
                for repr_name in repr_name_list:
                    repr_dict_clean[repr_name] = motion_repr_clean_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                    repr_dict_noisy[repr_name] = motion_repr_noisy_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                    repr_dict_rec[repr_name] = motion_repr_rec_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                    repr_dict_clean[repr_name] = torch.from_numpy(repr_dict_clean[repr_name]).to(dist_util.dev())
                    repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
                    repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
                    cur_total_dim += repr_dim_dict[repr_name]
                _, smpl_verts_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
                _, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
                _, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
                smpl_verts_clean = smpl_verts_clean[0].cpu().detach().numpy()
                smpl_verts_rec = smpl_verts_rec[0].cpu().detach().numpy()
                smpl_verts_noisy = smpl_verts_noisy[0].cpu().detach().numpy()

                for t in range(clip_len):
                    ############################################# body skeletons
                    cur_mask_joint_id = mask_joint_id.tolist() if args.mask_scheme == 'lower' or args.mask_scheme == 'video' else None
                    skeleton_gt_list = vis_skeleton(joints=rec_ric_data_clean_list[bs, t], limbs=LIMBS_BODY_SMPL, add_trans=np.array([0, 2.0, 2.5]))
                    skeleton_rec_list = vis_skeleton(joints=rec_ric_data_rec_list_from_smpl[bs, t], limbs=LIMBS_BODY_SMPL, add_trans=np.array([0, 0.0, 2.5]),
                                                     mask_scheme=args.mask_scheme, cur_mask_joint_id=cur_mask_joint_id)
                    if input_noise:
                        skeleton_noisy_list = vis_skeleton(joints=rec_ric_data_noisy_list[bs, t], limbs=LIMBS_BODY_SMPL, add_trans=np.array([0, 1.0, 2.5]),
                                                           mask_scheme=args.mask_scheme, cur_mask_joint_id=cur_mask_joint_id)

                    ############################################# foot contact labels
                    foot_sphere_clean_list = vis_foot_contact(joints=rec_ric_data_clean_list[bs, t], contact_lbl=contact_lbl_clean_list[bs, t], add_trans=np.array([0, 2.0, 0.0]))
                    foot_sphere_rec_list = vis_foot_contact(joints=rec_ric_data_rec_list_from_smpl[bs, t], contact_lbl=contact_lbl_rec_list[bs, t])

                    ############################################# body mesh
                    body_mesh_clean = o3d.geometry.TriangleMesh()
                    body_mesh_clean.vertices = o3d.utility.Vector3dVector(smpl_verts_clean[t])
                    body_mesh_clean.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                    body_mesh_clean.compute_vertex_normals()
                    body_mesh_clean.paint_uniform_color(COLOR_GT_O3D)
                    transformation = np.identity(4)
                    transformation[1, 3] = 2.0
                    body_mesh_clean.transform(transformation)

                    body_mesh_noisy = o3d.geometry.TriangleMesh()
                    body_mesh_noisy.vertices = o3d.utility.Vector3dVector(smpl_verts_noisy[t])
                    body_mesh_noisy.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                    body_mesh_noisy.compute_vertex_normals()
                    body_mesh_noisy.paint_uniform_color(COLOR_VIS_O3D)
                    transformation = np.identity(4)
                    transformation[1, 3] = 1.0
                    body_mesh_noisy.transform(transformation)

                    body_mesh_rec = o3d.geometry.TriangleMesh()
                    body_mesh_rec.vertices = o3d.utility.Vector3dVector(smpl_verts_rec[t])
                    body_mesh_rec.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                    body_mesh_rec.compute_vertex_normals()
                    body_mesh_rec.paint_uniform_color(COLOR_OCC_O3D)

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
                    if input_noise:
                        for arrow in skeleton_noisy_list:
                            vis.add_geometry(arrow)

                    ctr = vis.get_view_control()
                    cam_param = ctr.convert_to_pinhole_camera_parameters()
                    cam_param = update_cam(cam_param, cam_trans)
                    ctr.convert_from_pinhole_camera_parameters(cam_param)
                    vis.poll_events()
                    vis.update_renderer()
                    # time.sleep(0.2)

                    for sphere in foot_sphere_clean_list:
                        vis.remove_geometry(sphere)
                    for sphere in foot_sphere_rec_list:
                        vis.remove_geometry(sphere)
                    for arrow in skeleton_gt_list:
                        vis.remove_geometry(arrow)
                    for arrow in skeleton_rec_list:
                        vis.remove_geometry(arrow)
                    if input_noise:
                        for arrow in skeleton_noisy_list:
                            vis.remove_geometry(arrow)
                    vis.remove_geometry(body_mesh_clean)
                    vis.remove_geometry(body_mesh_noisy)
                    vis.remove_geometry(body_mesh_rec)

    ############################### render results
    if args.render:
        img_save_path_rec = os.path.join(args.render_save_path, 'pred')
        img_save_path_input = os.path.join(args.render_save_path, 'input')
        img_save_path_gt = os.path.join(args.render_save_path, 'gt')
        os.makedirs(img_save_path_rec) if not os.path.exists(img_save_path_rec) else None
        os.makedirs(img_save_path_input) if not os.path.exists(img_save_path_input) else None
        os.makedirs(img_save_path_gt) if not os.path.exists(img_save_path_gt) else None

        # common
        H, W = 1080, 1920
        camera, camera_pose, light = create_render_cam(cam_x=960, cam_y=540, fx=1060.53, fy=1060.38)
        cam_trans = np.array([[0, 0, -1, 5],
                              [-1, 0, 0, 1],
                              [0, -1, 0, 1],
                              [0, 0, 0, 1]])
        ground_mesh = create_floor(cam_trans)
        if args.mask_scheme == 'lower':
            cur_mask_joint_id = mask_joint_id.tolist()
            smplx_segment = json.load(open('data/smplx_vert_segmentation.json'))
            lower_body_verts_list = smplx_segment['leftLeg'] + smplx_segment['rightLeg'] + \
                                    smplx_segment['leftToeBase'] + smplx_segment['rightToeBase'] + \
                                    smplx_segment['leftFoot'] + smplx_segment['rightFoot'] + \
                                    smplx_segment['leftUpLeg'] + smplx_segment['rightUpLeg']
        elif args.mask_scheme == 'full':
            cur_mask_joint_id = None

        for bs in range(0, n_seq, args.render_interval):
            ################### get smplx vertices
            cur_total_dim = 0
            repr_dict_clean = {}
            repr_dict_rec = {}
            repr_dict_noisy = {}
            for repr_name in repr_name_list:
                repr_dict_clean[repr_name] = motion_repr_clean_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                repr_dict_noisy[repr_name] = motion_repr_noisy_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                repr_dict_rec[repr_name] = motion_repr_rec_list[[bs], ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                repr_dict_clean[repr_name] = torch.from_numpy(repr_dict_clean[repr_name]).to(dist_util.dev())
                repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
                repr_dict_noisy[repr_name] = torch.from_numpy(repr_dict_noisy[repr_name]).to(dist_util.dev())
                cur_total_dim += repr_dim_dict[repr_name]
            _, smpl_verts_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            _, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            _, smpl_verts_noisy = recover_from_repr_smpl(repr_dict_noisy, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True)
            smpl_verts_clean = smpl_verts_clean[0].cpu().detach().numpy()
            smpl_verts_rec = smpl_verts_rec[0].cpu().detach().numpy()
            smpl_verts_noisy = smpl_verts_noisy[0].cpu().detach().numpy()

            for t in range(0, clip_len, 1):
                scene_rec_body = create_pyrender_scene(camera, camera_pose, light)
                scene_noisy_body = create_pyrender_scene(camera, camera_pose, light)
                scene_gt_body = create_pyrender_scene(camera, camera_pose, light)
                scene_rec_skel = create_pyrender_scene(camera, camera_pose, light)
                scene_noisy_skel = create_pyrender_scene(camera, camera_pose, light)

                scene_rec_body.add(ground_mesh, 'mesh')
                scene_noisy_body.add(ground_mesh, 'mesh')
                scene_gt_body.add(ground_mesh, 'mesh')

                ################### add body mesh
                body_mesh_gt = create_pyrender_mesh(verts=smpl_verts_rec[t], faces=smplx_neutral.faces, trans=cam_trans, material=material_body_gt)
                if args.mask_scheme == 'lower' or (args.mask_scheme == 'full' and (t < start or t >= end)):
                    body_mesh_rec = create_pyrender_mesh(verts=smpl_verts_rec[t], faces=smplx_neutral.faces, trans=cam_trans, material=material_body_rec_vis)
                else:
                    body_mesh_rec = create_pyrender_mesh(verts=smpl_verts_rec[t], faces=smplx_neutral.faces, trans=cam_trans, material=material_body_rec_occ)
                if args.mask_scheme == 'lower':
                    vertex_colors = np.tile([198 / 255, 226 / 255, 255 / 255], (10475, 1))
                    vertex_alpha = np.ones((10475, 1))
                    vertex_alpha[lower_body_verts_list] = 0.1
                    vertex_colors = np.concatenate([vertex_colors, vertex_alpha], axis=-1)
                    body_mesh_noisy = create_pyrender_mesh(verts=smpl_verts_noisy[t], faces=smplx_neutral.faces, trans=cam_trans, vertex_colors=vertex_colors)
                else:
                    body_mesh_noisy = create_pyrender_mesh(verts=smpl_verts_noisy[t], faces=smplx_neutral.faces, trans=cam_trans, material=material_body_noisy)

                scene_rec_body.add(body_mesh_rec, 'mesh')
                scene_noisy_body.add(body_mesh_noisy, 'mesh')
                scene_gt_body.add(body_mesh_gt, 'mesh')

                ################## add body skeleton
                skeleton_mesh_rec_list = create_pyrender_skel(joints=rec_ric_data_rec_list_from_smpl[bs, t], add_trans=np.linalg.inv(cam_trans),
                                                              mask_scheme=args.mask_scheme, mask_joint_id=cur_mask_joint_id, add_occ_joints=True,
                                                              add_contact=True, t=t, start=start, end=end, contact_lbl=contact_lbl_rec_list[bs, t])
                skeleton_mesh_noisy_list = create_pyrender_skel(joints=rec_ric_data_noisy_list[bs, t], add_trans=np.linalg.inv(cam_trans), mask_scheme=args.mask_scheme,
                                                                mask_joint_id=cur_mask_joint_id, add_occ_joints=False, t=t, start=start, end=end, add_contact=False,)
                for mesh in skeleton_mesh_rec_list:
                    scene_rec_skel.add(mesh, 'pred_joint')
                for mesh in skeleton_mesh_noisy_list:
                    scene_noisy_skel.add(mesh, 'input_joint')

                ################## render images
                r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
                ####### render: pred body
                color_rec_body = render_img(r, scene_rec_body, alpha=1.0)
                color_rec_skel = render_img(r, scene_rec_skel, alpha=1.0)
                color_rec_body.paste(color_rec_skel, (0, 0), color_rec_skel)
                color_rec_body = color_rec_body.transpose(Image.FLIP_LEFT_RIGHT)
                img_save_path_rec_cur = os.path.join(img_save_path_rec, 'seq_{}'.format(format(bs, '03d')))
                os.makedirs(img_save_path_rec_cur) if not os.path.exists(img_save_path_rec_cur) else None
                color_rec_body.save(os.path.join(img_save_path_rec_cur, 'frame_{}.png'.format(format(t, '03d'))))

                ####### render: noisy body
                alpha = 1.0
                if args.mask_scheme == 'full':
                    if t >= start and t < end:
                        alpha = 0.5
                color_noisy_body = render_img(r, scene_noisy_body, alpha=alpha)
                color_noisy_skel = render_img(r, scene_noisy_skel, alpha=alpha)
                color_noisy_body.paste(color_noisy_skel, (0, 0), color_noisy_skel)
                color_noisy_body = color_noisy_body.transpose(Image.FLIP_LEFT_RIGHT)
                img_save_path_input_cur = os.path.join(img_save_path_input, 'seq_{}'.format(format(bs, '03d')))
                os.makedirs(img_save_path_input_cur) if not os.path.exists(img_save_path_input_cur) else None
                color_noisy_body.save(os.path.join(img_save_path_input_cur, 'frame_{}.png'.format(format(t, '03d'))))

                ####### render: gt body
                color_gt_body = render_img(r, scene_gt_body, alpha=1.0)
                color_gt_body = color_gt_body.transpose(Image.FLIP_LEFT_RIGHT)
                img_save_path_gt_cur = os.path.join(img_save_path_gt, 'seq_{}'.format(format(bs, '03d')))
                os.makedirs(img_save_path_gt_cur) if not os.path.exists(img_save_path_gt_cur) else None
                color_gt_body.save(os.path.join(img_save_path_gt_cur, 'frame_{}.png'.format(format(t, '03d'))))


