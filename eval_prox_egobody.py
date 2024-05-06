import pickle
from data_loaders.motion_representation import *
from utils import dist_util
from utils.vis_util import *
from utils.render_util import *
import smplx
import pandas as pd
from tqdm import tqdm
import configargparse
import cv2
import PIL.Image as pil_img
import pyrender

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
group.add_argument('--dataset', type=str, default='egobody', choices=['prox', 'egobody'])
group.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to dataset')
group.add_argument('--saved_data_dir', type=str,
                   default='data/test_results_release/results_egobody_rgb/test_egobody_grad_True_iter_2_iter2trajnoisy_False_iter2posenoisy_False_earlystop_True_seed_0',  #
                   help='path to saved test results')
group.add_argument('--recording_name', type=str, default='recording_20210907_S02_S01_01', help='all - evaluate on all subsequences; otherwise specify the recording name to evaluate/visualize')

group.add_argument('--visualize', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument('--vis_option', default='mesh', type=str, choices=['mesh', 'skeleton'])
group.add_argument("--vis_interval", default=1, type=int, help="visualize every N clips")

group.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'])
group.add_argument("--render_interval", default=100, type=int, help="render every N clips")
group.add_argument("--render_save_path", default='render_imgs/render_egobody_rgb', type=str, help='path to save render images')



args = group.parse_args()
dist_util.setup_dist(args.device)
smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                             gender='neutral', flat_hand_mean=True, use_pca=False).to(dist_util.dev())

if __name__ == "__main__":
    if args.visualize and args.render:
        print('[ERROR] cannot visualize and render at the same time.')
        exit()

    if args.recording_name != 'all':
        test_recording_name_list = [args.recording_name]
    else:
        if args.dataset == 'prox':
            test_recording_name_list = \
                ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
                 'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
                 'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
                 'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
                 'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']
        elif args.dataset == 'egobody':
            test_recording_name_list = \
                ['recording_20210907_S02_S01_01', 'recording_20210907_S03_S04_01', 'recording_20210929_S05_S16_01',
                 'recording_20210929_S05_S16_04', 'recording_20211004_S19_S06_01', 'recording_20211004_S19_S06_02',
                 'recording_20211004_S19_S06_03', 'recording_20211004_S12_S20_01', 'recording_20211004_S12_S20_02',
                 'recording_20211004_S12_S20_03', 'recording_20220315_S21_S30_03', 'recording_20220315_S21_S30_05',
                 'recording_20220318_S32_S31_01', 'recording_20220318_S32_S31_02', 'recording_20220318_S34_S33_01',
                 'recording_20220318_S33_S34_01', 'recording_20220318_S33_S34_02', 'recording_20220415_S36_S35_02',
                 'recording_20220415_S35_S36_02']
        else:
            test_recording_name_list = None

    ################################# read egobody data info
    if args.dataset == 'egobody':
        df = pd.read_csv(os.path.join(args.dataset_root, 'egobody_rohm_info.csv'))
        recording_name_list = list(df['recording_name'])
        start_frame_list = list(df['target_start_frame'])
        end_frame_list = list(df['target_end_frame'])
        idx_list = list(df['target_idx'])
        gender_list = list(df['target_gender'])
        view_list = list(df['view'])
        scene_name_list = list(df['scene_name'])
        body_idx_fpv_list = list(df['body_idx_fpv'])

        start_frame_dict = dict(zip(recording_name_list, start_frame_list))
        end_frame_dict = dict(zip(recording_name_list, end_frame_list))
        idx_dict = dict(zip(recording_name_list, idx_list))
        gender_dict = dict(zip(recording_name_list, gender_list))
        view_dict = dict(zip(recording_name_list, view_list))
        scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))

    if args.visualize:
        import open3d as o3d
        from utils.other_utils import LIMBS_BODY_SMPL
        from utils.other_utils import *
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_frame)
        print('Visualizing...')
        if args.vis_option == 'skeleton':
            print('[blue/yellow - prediction] [blue] visible parts / [yellow] occluded parts')
            print('[green - initialized input]')
            print('[foot contact label - prediction]: [red] not in contact with floor / [green] in contact with floor')
        elif args.vis_option == 'mesh':
            print('[blue - prediction]')
            print('[green - initialized input]')

    ################################# evaluate metrics
    skating_list = {}
    acc_list = {}
    acc_error_list = {}
    ground_pene_dist_list = {}
    ground_pene_freq_list = {}
    gmpjpe_list = {}
    mpjpe_list = {}
    mpjpe_list_vis = {}
    mpjpe_list_occ = {}
    joint_mask_list = {}
    for recording_name in test_recording_name_list:
        if args.dataset == 'prox':
            cam2world_dir = os.path.join(args.dataset_root, 'cam2world')
            scene_name = recording_name.split("_")[0]
            with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
                cam2world = np.array(json.load(f))
        elif args.dataset == 'egobody':
            view = view_dict[recording_name]
            body_idx = idx_dict[recording_name]
            scene_name = scene_name_dict[recording_name]
            gender_gt = gender_dict[recording_name]
            ######################### load calibration from sub kinect to main kinect (between color cameras)
            # master: kinect 12, sub_1: kinect 11, sub_2: kinect 13, sub_3, kinect 14, sub_4: kinect 15
            calib_trans_dir = os.path.join(args.dataset_root, 'calibrations', recording_name)  # extrinsics
            with open(os.path.join(calib_trans_dir, 'cal_trans', 'kinect12_to_world', scene_name + '.json'), 'r') as f:
                cam2world = np.asarray(json.load(f)['trans'])
            if view == 'sub_1':
                trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_11to12_color.json')
            elif view == 'sub_2':
                trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_13to12_color.json')
            elif view == 'sub_3':
                trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_14to12_color.json')
            elif view == 'sub_4':
                trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_15to12_color.json')
            if view != 'master':
                with open(os.path.join(trans_subtomain_path), 'r') as f:
                    trans_subtomain = np.asarray(json.load(f)['trans'])
                cam2world = np.matmul(cam2world, trans_subtomain)

        ################################# read test results data
        saved_data_path = '{}/{}.pkl'.format(args.saved_data_dir, recording_name)
        with open(saved_data_path, 'rb') as f:
            saved_data = pickle.load(f)
        print(saved_data_path)
        repr_name_list = saved_data['repr_name_list']
        repr_dim_dict = saved_data['repr_dim_dict']
        frame_name_list = saved_data['frame_name_list'] if args.dataset == 'egobody' else None
        rec_ric_data_noisy_list = saved_data['rec_ric_data_noisy_list']
        joints_gt_scene_coord_list = saved_data['joints_gt_scene_coord_list'] if args.dataset == 'egobody' else None
        rec_ric_data_rec_list_from_smpl = saved_data['rec_ric_data_rec_list_from_smpl']
        joints_input_scene_coord_list = saved_data['joints_input_scene_coord_list']
        motion_repr_rec_list = saved_data['motion_repr_rec_list']
        motion_repr_noisy_list = saved_data['motion_repr_noisy_list']
        mask_joint_vis_list = saved_data['mask_joint_vis_list']  # [n_clip, 143, 22]
        trans_scene2cano_list = saved_data['trans_scene2cano_list']
        n_seq = len(rec_ric_data_noisy_list)
        clip_len_rec = rec_ric_data_noisy_list.shape[1]
        print('n_seq:', n_seq)
        print('clip_len_rec:', clip_len_rec)
        joints_gt_scene_coord_list = joints_gt_scene_coord_list[:, 0:clip_len_rec] if args.dataset == 'egobody' else None

        ################ get contact lbls
        contact_lbl_rec_list = motion_repr_rec_list[:, :, -4:]  # np, [n_seq, clip_len, 4]
        contact_lbl_rec_list[contact_lbl_rec_list > 0.5] = 1.0
        contact_lbl_rec_list[contact_lbl_rec_list <= 0.5] = 0.0

        ################### transform back to scene coord
        for seq_idx in range(n_seq):
            cur_joints_scene_coord = points_coord_trans(rec_ric_data_noisy_list[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
            rec_ric_data_noisy_list[seq_idx] = cur_joints_scene_coord.reshape(clip_len_rec, 22, 3)
            cur_joints_scene_coord = points_coord_trans(rec_ric_data_rec_list_from_smpl[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
            rec_ric_data_rec_list_from_smpl[seq_idx] = cur_joints_scene_coord.reshape(clip_len_rec, 22, 3)

        ############################### skating ratio
        thresh_height = 0.10
        thresh_vel = 0.10
        fps = 30
        foot_joint_index_list = [7, 10, 8, 11]  # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
        joints_foot_rec = rec_ric_data_rec_list_from_smpl[:, :, foot_joint_index_list, :]  # [n_seq, clip_len, 2, 3]
        if args.dataset == 'prox':
            # prox scene coord up axis is z
            ground_height = prox_floor_height[scene_name]
            joints_feet_horizon_vel_rec = np.linalg.norm(joints_foot_rec[:, 1:, :, [0, 1]] - joints_foot_rec[:, :-1, :, [0, 1]], axis=-1) * fps  # [n_seq, clip_len, 2]
            joints_feet_height_rec = joints_foot_rec[:, 0:-1, :, 2]  # [n_seq, clip_len, 2]
        elif args.dataset == 'egobody':
            # egobody scene coord up axis is y
            ground_height = egobody_floor_height[scene_name]
            joints_feet_horizon_vel_rec = np.linalg.norm(joints_foot_rec[:, 1:, :, [0, 2]] - joints_foot_rec[:, :-1, :, [0, 2]], axis=-1) * fps
            joints_feet_height_rec = joints_foot_rec[:, 0:-1, :, 1]
        joints_feet_height_rec = joints_feet_height_rec - ground_height
        skating_rec_left = (joints_feet_horizon_vel_rec[:, :, 0] > thresh_vel) * (joints_feet_horizon_vel_rec[:, :, 1] > thresh_vel) * \
                           (joints_feet_height_rec[:, :, 0] < (thresh_height + 0.05)) * (joints_feet_height_rec[:, :, 1] < thresh_height)
        skating_rec_right = (joints_feet_horizon_vel_rec[:, :, 2] > thresh_vel) * (joints_feet_horizon_vel_rec[:, :, 3] > thresh_vel) * \
                            (joints_feet_height_rec[:, :, 2] < (thresh_height + 0.05)) * (joints_feet_height_rec[:, :, 3] < thresh_height)
        skating_rec = skating_rec_left * skating_rec_right  # [n_clip, 142]
        if recording_name not in skating_list.keys():
            skating_list[recording_name] = []
            skating_list[recording_name].append(skating_rec)
        else:
            skating_list[recording_name].append(skating_rec)

        ########################### acceleration metrics
        acc_rec = (rec_ric_data_rec_list_from_smpl[:, 2:] - 2 * rec_ric_data_rec_list_from_smpl[:, 1:-1] + rec_ric_data_rec_list_from_smpl[:, :-2]) * (fps ** 2)  # [n_clip, 141, 22, 3]
        if args.dataset == 'egobody':
            acc_gt = (joints_gt_scene_coord_list[:, 2:] - 2 * joints_gt_scene_coord_list[:, 1:-1] + joints_gt_scene_coord_list[:, :-2]) * (fps ** 2)
            acc_error = np.linalg.norm(acc_rec - acc_gt, axis=-1).mean(axis=-1)
        acc_rec = np.linalg.norm(acc_rec, axis=-1).mean(axis=-1)  # [n_clip, 141]
        if recording_name not in acc_error_list.keys():
            acc_list[recording_name] = []
            acc_list[recording_name].append(acc_rec)
            if args.dataset == 'egobody':
                acc_error_list[recording_name] = []
                acc_error_list[recording_name].append(acc_error)
        else:
            acc_list[recording_name].append(acc_rec)
            acc_error_list[recording_name].append(acc_error) if args.dataset == 'egobody' else None

        ########################### mpjpe metrics
        if args.dataset == 'egobody':
            if recording_name not in joint_mask_list.keys():
                joint_mask_list[recording_name] = []
                joint_mask_list[recording_name].append(mask_joint_vis_list)
            else:
                joint_mask_list[recording_name].append(mask_joint_vis_list)

            joints_mpjpe_global = np.linalg.norm(joints_gt_scene_coord_list - rec_ric_data_rec_list_from_smpl, axis=-1)  # [n_seq, clip_len, 22]
            joints_mpjpe_local = np.linalg.norm((joints_gt_scene_coord_list - joints_gt_scene_coord_list[:, 0:clip_len_rec, [0]]) -
                                                (rec_ric_data_rec_list_from_smpl - rec_ric_data_rec_list_from_smpl[:, :, [0]]), axis=-1)
            joints_mpjpe_local_vis = joints_mpjpe_local * mask_joint_vis_list
            joints_mpjpe_local_invis = joints_mpjpe_local * (1 - mask_joint_vis_list)
            if recording_name not in gmpjpe_list.keys():
                gmpjpe_list[recording_name] = []
                gmpjpe_list[recording_name].append(joints_mpjpe_global)
                mpjpe_list[recording_name] = []
                mpjpe_list_vis[recording_name] = []
                mpjpe_list_occ[recording_name] = []
                mpjpe_list[recording_name].append(joints_mpjpe_local)
                mpjpe_list_vis[recording_name].append(joints_mpjpe_local_vis)
                mpjpe_list_occ[recording_name].append(joints_mpjpe_local_invis)
            else:
                gmpjpe_list[recording_name].append(joints_mpjpe_global)
                mpjpe_list[recording_name].append(joints_mpjpe_local)
                mpjpe_list_vis[recording_name].append(joints_mpjpe_local_vis)
                mpjpe_list_occ[recording_name].append(joints_mpjpe_local_invis)

        ########################### ground penetration metrics
        if args.dataset == 'egobody':
            pene_dist = rec_ric_data_rec_list_from_smpl[:, :, [10, 11], 1] - ground_height  # [n_clip, 143, 2]
        elif args.dataset == 'prox':
            pene_dist = rec_ric_data_rec_list_from_smpl[:, :, [10, 11], 2] - ground_height
        pene_freq = pene_dist < -0.05  # [clip_len]
        pene_freq = pene_freq.mean(axis=-1)  # [n_clip, 143]
        pene_dist[pene_dist >= 0] = 0
        pene_dist = pene_dist.mean(axis=-1)  # [n_clip, 143]
        if recording_name not in ground_pene_dist_list.keys():
            ground_pene_dist_list[recording_name] = []
            ground_pene_freq_list[recording_name] = []
            ground_pene_dist_list[recording_name].append(pene_dist)
            ground_pene_freq_list[recording_name].append(pene_freq)
        else:
            ground_pene_dist_list[recording_name].append(pene_dist)
            ground_pene_freq_list[recording_name].append(pene_freq)

        if args.visualize or args.render:
            ############ get smplx vertices
            smpl_verts_rec_list = []
            joints_rec_list = []
            smpl_verts_input_list = []
            joints_input_list = []
            with torch.no_grad():
                for idx in tqdm(range(n_seq)):
                    cur_total_dim = 0
                    repr_dict_rec = {}
                    repr_dict_input = {}
                    for repr_name in repr_name_list:
                        repr_dict_rec[repr_name] = motion_repr_rec_list[idx:(idx + 1), ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                        repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
                        repr_dict_input[repr_name] = motion_repr_noisy_list[idx:(idx + 1), ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                        repr_dict_input[repr_name] = torch.from_numpy(repr_dict_input[repr_name]).to(dist_util.dev())
                        cur_total_dim += repr_dim_dict[repr_name]
                    joints_rec, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True, return_full_joints=True)
                    joints_input, smpl_verts_input = recover_from_repr_smpl(repr_dict_input, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True, return_full_joints=True)
                    smpl_verts_rec_list.append(smpl_verts_rec.detach().cpu().numpy())
                    joints_rec_list.append(joints_rec.detach().cpu().numpy())
                    smpl_verts_input_list.append(smpl_verts_input.detach().cpu().numpy())
                    joints_input_list.append(joints_input.detach().cpu().numpy())
            smpl_verts_rec_list = np.concatenate(smpl_verts_rec_list, axis=0)
            joints_rec_list = np.concatenate(joints_rec_list, axis=0)
            smpl_verts_input_list = np.concatenate(smpl_verts_input_list, axis=0)
            joints_input_list = np.concatenate(joints_input_list, axis=0)
            ########### transform back to scene coord
            for seq_idx in range(n_seq):
                cur_verts_scene_coord = points_coord_trans(smpl_verts_rec_list[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
                smpl_verts_rec_list[seq_idx] = cur_verts_scene_coord.reshape(clip_len_rec, -1, 3)
                cur_joints_scene_coord = points_coord_trans(joints_rec_list[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
                joints_rec_list[seq_idx] = cur_joints_scene_coord.reshape(clip_len_rec, -1, 3)
                cur_verts_scene_coord = points_coord_trans(smpl_verts_input_list[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
                smpl_verts_input_list[seq_idx] = cur_verts_scene_coord.reshape(clip_len_rec, -1, 3)
                cur_joints_scene_coord = points_coord_trans(joints_input_list[seq_idx].reshape(-1, 3), np.linalg.inv(trans_scene2cano_list[seq_idx]))
                joints_input_list[seq_idx] = cur_joints_scene_coord.reshape(clip_len_rec, -1, 3)

        ####################################### visualization #############################
        if args.visualize:
            for bs in range(0, n_seq, 1):
                if bs % args.vis_interval == 0:
                    for t in range(0, clip_len_rec, 1):
                        ################################# body skeletons
                        cur_joint_mask_vis = mask_joint_vis_list[bs, t]  # [22]
                        cur_mask_joint_id = np.where(cur_joint_mask_vis == 0)[0].tolist()
                        skeleton_input_list = vis_skeleton(joints=joints_input_list[bs, t], limbs=LIMBS_BODY_SMPL,
                                                           mask_scheme='video', cur_mask_joint_id=cur_mask_joint_id,
                                                           color_occ=[0, 128 / 255, 0], color_vis=[0, 128 / 255, 0])
                        skeleton_rec_list = vis_skeleton(joints=joints_rec_list[bs, t], limbs=LIMBS_BODY_SMPL,
                                                         mask_scheme='video', cur_mask_joint_id=cur_mask_joint_id)

                        ################################# foot contact labels
                        foot_sphere_rec_list = vis_foot_contact(joints=joints_rec_list[bs, t], contact_lbl=contact_lbl_rec_list[bs, t])

                        ################################# body mesh
                        body_mesh_rec = o3d.geometry.TriangleMesh()
                        body_mesh_rec.vertices = o3d.utility.Vector3dVector(smpl_verts_rec_list[bs, t])
                        body_mesh_rec.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                        body_mesh_rec.compute_vertex_normals()
                        body_mesh_rec.paint_uniform_color(COLOR_VIS_O3D)

                        body_mesh_input = o3d.geometry.TriangleMesh()
                        body_mesh_input.vertices = o3d.utility.Vector3dVector(smpl_verts_input_list[bs, t])
                        body_mesh_input.triangles = o3d.utility.Vector3iVector(smplx_neutral.faces)
                        body_mesh_input.compute_vertex_normals()
                        body_mesh_input.paint_uniform_color([0, 128 / 255, 0])

                        if args.vis_option == 'mesh':
                            vis.add_geometry(body_mesh_rec)
                            vis.add_geometry(body_mesh_input)
                        if args.vis_option == 'skeleton':
                            for arrow in skeleton_rec_list:
                                vis.add_geometry(arrow)
                            for arrow in skeleton_input_list:
                                vis.add_geometry(arrow)
                        for sphere in foot_sphere_rec_list:
                            vis.add_geometry(sphere)

                        ctr = vis.get_view_control()
                        cam_param = ctr.convert_to_pinhole_camera_parameters()
                        cam_param = update_cam(cam_param, cam2world)
                        ctr.convert_from_pinhole_camera_parameters(cam_param)
                        vis.poll_events()
                        vis.update_renderer()
                        # time.sleep(0.03)

                        if args.vis_option == 'mesh':
                            vis.remove_geometry(body_mesh_rec)
                            vis.remove_geometry(body_mesh_input)
                        if args.vis_option == 'skeleton':
                            for arrow in skeleton_rec_list:
                                vis.remove_geometry(arrow)
                            for arrow in skeleton_input_list:
                                vis.remove_geometry(arrow)
                        for sphere in foot_sphere_rec_list:
                            vis.remove_geometry(sphere)

        ####################################### render results #############################
        if args.render:
            img_save_path_mesh_skel_rec = os.path.join(args.render_save_path, 'mesh_skel')
            img_save_path_mesh_noisy = os.path.join(args.render_save_path, 'input')
            os.makedirs(img_save_path_mesh_skel_rec) if not os.path.exists(img_save_path_mesh_skel_rec) else None
            os.makedirs(img_save_path_mesh_noisy) if not os.path.exists(img_save_path_mesh_noisy) else None

            H, W = 1080, 1920
            ########## read kinect color camera intrinsics
            if args.dataset == 'egobody':
                with open(os.path.join(args.dataset_root, 'kinect_cam_params', 'kinect_{}'.format(view), 'Color.json'), 'r') as f:
                    color_cam = json.load(f)
            elif args.dataset == 'prox':
                with open(os.path.join(args.dataset_root, 'calibration', 'Color.json'), 'r') as f:
                    color_cam = json.load(f)
            [f_x, f_y] = color_cam['f']
            [c_x, c_y] = color_cam['c']
            camera, camera_pose, light = create_render_cam(cam_x=c_x, cam_y=c_y, fx=f_x, fy=f_y)

            if args.dataset == 'egobody':
                rgb_img_root = os.path.join(args.dataset_root, 'kinect_color', recording_name, view)
            elif args.dataset == 'prox':
                rgb_img_root = os.path.join(args.dataset_root, 'recordings', recording_name, 'Color')
                rgb_frame_list = os.listdir(rgb_img_root)
                rgb_frame_list = sorted(rgb_frame_list)
                img_frame_idx = 0  # 0
            print('[INFO] saving images...')
            for bs in tqdm(range(0, n_seq, 1)):
                for t in range(0, clip_len_rec, 1):
                    if args.dataset == 'egobody':
                        img_path = os.path.join(rgb_img_root, frame_name_list[bs, t] + '.jpg')
                    elif args.dataset == 'prox':
                        img_path = os.path.join(rgb_img_root, rgb_frame_list[img_frame_idx])
                    cur_img = cv2.imread(img_path)
                    cur_img = cur_img[:, :, ::-1]
                    if args.dataset == 'prox':
                        cur_img = cv2.undistort(cur_img.copy(), np.asarray(color_cam['camera_mtx']), np.asarray(color_cam['k']))
                        cur_img = cv2.flip(cur_img, 1)

                    ########## read joint visibility mask
                    cur_joint_mask_vis = mask_joint_vis_list[bs, t]  # [22]
                    cur_mask_joint_id = np.where(cur_joint_mask_vis == 0)[0].tolist()

                    ########## create pyrender scenes
                    scene_rec_body = create_pyrender_scene(camera, camera_pose, light)
                    scene_noisy_body = create_pyrender_scene(camera, camera_pose, light)
                    scene_rec_skel = create_pyrender_scene(camera, camera_pose, light)
                    scene_noisy_skel = create_pyrender_scene(camera, camera_pose, light)

                    ################### add body mesh
                    body_mesh_rec = create_pyrender_mesh(verts=smpl_verts_rec_list[bs, t], faces=smplx_neutral.faces, trans=cam2world, material=material_body_rec_vis)
                    body_mesh_input = create_pyrender_mesh(verts=smpl_verts_input_list[bs, t], faces=smplx_neutral.faces, trans=cam2world, material=material_body_noisy)
                    scene_rec_body.add(body_mesh_rec, 'mesh')
                    scene_noisy_body.add(body_mesh_input, 'mesh')

                    ################## add body skeleton
                    skeleton_mesh_rec_list = create_pyrender_skel(joints=rec_ric_data_rec_list_from_smpl[bs, t], add_trans=np.linalg.inv(cam2world),
                                                                  mask_scheme='video', mask_joint_id=cur_mask_joint_id,
                                                                  add_contact=True, contact_lbl=contact_lbl_rec_list[bs, t])
                    for mesh in skeleton_mesh_rec_list:
                        scene_rec_skel.add(mesh, 'pred_joint')

                    ################## render images
                    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
                    ####### render: pred body
                    img_rec_body = render_img(r, scene_rec_body, alpha=0.9)
                    img_rec_skel = render_img(r, scene_rec_skel, alpha=1.0)
                    render_img_input = render_img_overlay(r, scene_noisy_body, cur_img)

                    render_img_rec = pil_img.fromarray((cur_img).astype(np.uint8))
                    render_img_rec.paste(img_rec_body, (0, 0), img_rec_body)
                    render_img_rec.paste(img_rec_skel, (0, 0), img_rec_skel)

                    if args.dataset == 'egobody':
                        render_img_input.save(os.path.join(img_save_path_mesh_noisy, frame_name_list[bs, t] + '.jpg'))
                        render_img_rec.save(os.path.join(img_save_path_mesh_skel_rec, frame_name_list[bs, t] + '.jpg'))
                    elif args.dataset == 'prox':
                        render_img_input.save(os.path.join(img_save_path_mesh_noisy, rgb_frame_list[img_frame_idx]))
                        render_img_rec.save(os.path.join(img_save_path_mesh_skel_rec, rgb_frame_list[img_frame_idx]))
                        img_frame_idx += 1

    ########################################### final metrics ###############################################
    for recording_name in test_recording_name_list:
        skating_list[recording_name] = np.concatenate(skating_list[recording_name], axis=0)
        acc_list[recording_name] = np.concatenate(acc_list[recording_name], axis=0)
        if args.dataset == 'egobody':
            acc_error_list[recording_name] = np.concatenate(acc_error_list[recording_name], axis=0)
            joint_mask_list[recording_name] = np.concatenate(joint_mask_list[recording_name], axis=0)
            gmpjpe_list[recording_name] = np.concatenate(gmpjpe_list[recording_name], axis=0)
            mpjpe_list[recording_name] = np.concatenate(mpjpe_list[recording_name], axis=0)
            mpjpe_list_vis[recording_name] = np.concatenate(mpjpe_list_vis[recording_name], axis=0)
            mpjpe_list_occ[recording_name] = np.concatenate(mpjpe_list_occ[recording_name], axis=0)
        ground_pene_freq_list[recording_name] = np.concatenate(ground_pene_freq_list[recording_name], axis=0)
        ground_pene_dist_list[recording_name] = np.concatenate(ground_pene_dist_list[recording_name], axis=0)

    print('\n --------------- evaluation metrics -------------')
    skating_list['all'] = np.concatenate([skating_list[recording_name] for recording_name in test_recording_name_list], axis=0)
    acc_list['all'] = np.concatenate([acc_list[recording_name] for recording_name in test_recording_name_list], axis=0)
    if args.dataset == 'egobody':
        acc_error_list['all'] = np.concatenate([acc_error_list[recording_name] for recording_name in test_recording_name_list], axis=0)
        joint_mask_list['all'] = np.concatenate([joint_mask_list[recording_name] for recording_name in test_recording_name_list], axis=0)
        gmpjpe_list['all'] = np.concatenate([gmpjpe_list[recording_name] for recording_name in test_recording_name_list], axis=0)
        mpjpe_list['all'] = np.concatenate([mpjpe_list[recording_name] for recording_name in test_recording_name_list], axis=0)
        mpjpe_list_vis['all'] = np.concatenate([mpjpe_list_vis[recording_name] for recording_name in test_recording_name_list], axis=0)
        mpjpe_list_occ['all'] = np.concatenate([mpjpe_list_occ[recording_name] for recording_name in test_recording_name_list], axis=0)
    ground_pene_freq_list['all'] = np.concatenate([ground_pene_freq_list[recording_name] for recording_name in test_recording_name_list], axis=0)
    ground_pene_dist_list['all'] = np.concatenate([ground_pene_dist_list[recording_name] for recording_name in test_recording_name_list], axis=0)

    print('skating score: {:0.3f}'.format(skating_list['all'].mean()))
    print('||acc|| (m/s^2): {:0.2f}'.format(acc_list['all'].mean())) if args.dataset == 'prox' else None
    print('acc errors (m/s^2): {:0.2f}'.format(acc_error_list['all'].mean())) if args.dataset == 'egobody' else None
    print('ground_pene_freq score (%): {:0.2f}'.format(ground_pene_freq_list['all'].mean()*100))
    print('ground_pene_dist score (mm): {:0.2f}'.format(-ground_pene_dist_list['all'].mean()*1000))
    if args.dataset == 'egobody':
        print('-------------- gmpjpe/mpjpe/mpjpe-vis/mpjpe-occ (mm) --------------')
        print('{:0.2f} / {:0.2f} / {:0.2f} / {:0.2f}'.
              format(gmpjpe_list['all'].mean() * 1000, mpjpe_list['all'].mean() * 1000,
                     mpjpe_list_vis['all'].sum() / ((joint_mask_list['all']).sum()) * 1000,
                     mpjpe_list_occ['all'].sum() / ((1 - joint_mask_list['all']).sum()) * 1000))

