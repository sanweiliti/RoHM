from torch.utils import data
from tqdm import tqdm
import smplx
from data_loaders.motion_representation import *
import pickle as pkl
import pandas as pd
from utils.other_utils import *



class DataloaderVideo(data.Dataset):
    def __init__(self,
                 dataset='prox',
                 init_root='',
                 base_dir='',
                 body_model_path='',
                 recording_name='',
                 use_scene_floor_height=False,
                 repr_abs_only=False,
                 task='traj',
                 overlap_len=2,
                 clip_len=150, joints_num=22,
                 logdir=None, device='cpu'):
        self.dataset = dataset
        if self.dataset not in ['prox', 'egobody']:
            print('[ERROR] {} not defined.'.format(dataset))
        self.clip_len = clip_len
        self.clip_overlap_len = overlap_len  # overlapping window between each two clips
        self.logdir = logdir
        self.device = device
        self.smplx_neutral = smplx.create(model_path=body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)
        if self.dataset == 'egobody':
            # gender models to load ground truth body in EgoBody
            self.smplx_male = smplx.create(model_path=body_model_path, model_type="smplx",
                                           gender='male', flat_hand_mean=True, use_pca=False).to(self.device)
            self.smplx_female = smplx.create(model_path=body_model_path, model_type="smplx",
                                             gender='female', flat_hand_mean=True, use_pca=False).to(self.device)

        self.init_root = init_root
        self.base_dir = base_dir
        self.recording_name = recording_name
        self.with_smpl = True
        self.use_scene_floor_height = use_scene_floor_height

        self.joints_num = joints_num
        self.head_joint_idx = [15]  # smpl joint 15: chin actually
        self.torso_joint_idx = [12, 9, 6, 3]  # neck, spine
        # openpose joint 8 - smpl joint 0, for 24 smpl joints
        self.openpose_to_smpl = [8, 12, 9, 8, 13, 10, 8, 14, 11, 1, 20, 23, 1, 5, 2, 0, 5, 2, 6, 3, 7, 4, 7, 4][0:self.joints_num]

        self.task = task
        if self.task not in ['traj', 'pose']:
            print('[ERROR] args.traj should be in [traj, pose]!')
            exit()

        ########################################## configs for motion representation
        self.repr_abs_only = repr_abs_only  # if True, only include absolute repr for traj repr (exclude traj velocities)
        if not repr_abs_only:
            self.traj_repr_name_list = ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel',
                                        'root_height',
                                        'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel']
        else:
            self.traj_repr_name_list = ['root_rot_angle', 'root_l_pos', 'root_height',
                                        'smplx_rot_6d', 'smplx_trans']
        self.local_repr_name_list = ['local_positions', 'local_vel',
                                     'smplx_body_pose_6d', 'smplx_betas', 'foot_contact', ]

        ## get dimensions for features
        self.body_feat_dim = 0
        self.traj_feat_dim = 0
        self.pose_feat_dim = 0
        for repr_name in REPR_LIST:
            self.body_feat_dim += REPR_DIM_DICT[repr_name]
            if repr_name in self.traj_repr_name_list:
                self.traj_feat_dim += REPR_DIM_DICT[repr_name]
            if repr_name in self.local_repr_name_list:
                self.pose_feat_dim += REPR_DIM_DICT[repr_name]

        self.repr_list_input_dict = {}
        for repr_name in REPR_LIST:
            self.repr_list_input_dict[repr_name] = []

        self.cano_smplx_params_dict_list = []
        self.cano_joints_input_list = []
        self.transf_matrix_list = []  # transformation matrix from scene/world coord to canonicalized coord

        if self.dataset == 'prox':
            self.read_data_prox()
        if self.dataset == 'egobody':
            self.read_data_egobody()
        self.create_body_repr()


    def read_data_prox(self):
        fitting_dir = os.path.join(self.init_root, self.recording_name, 'results')
        scene_name = self.recording_name.split("_")[0]
        joint_mask_path = os.path.join(self.base_dir, 'mask_joint', self.recording_name, 'mask_joint.npy')
        self.scene_floor_height = prox_floor_height[scene_name]
        with open(os.path.join(self.base_dir, 'cam2world', scene_name + '.json'), 'r') as f:
            cam2world = np.array(json.load(f))
        cam2world = torch.from_numpy(cam2world).float().to(self.device)
        self.cam_R = cam2world[:3, :3].reshape([3, 3])
        self.cam_t = cam2world[:3, 3].reshape([1, 3])
        with open(os.path.join(self.base_dir, 'calibration', 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)

        frame_list = os.listdir(fitting_dir)
        frame_list.sort()
        frame_total = len(frame_list)
        print('[INFO] total frames of current sequence: ', frame_total)
        frame_name_list = []
        joints_list_world = []
        smplx_list_world = []
        keypoints_2d_list = []
        for cur_frame_name in tqdm(frame_list):
            frame_name_list.append(cur_frame_name)
            #### load current frame initialized smplx params
            cur_prox_params_dir = os.path.join(fitting_dir, cur_frame_name, '000.pkl')
            with open(cur_prox_params_dir, 'rb') as f:
                param = pkl.load(f)
            torch_param = {}
            torch_param['transl'] = torch.tensor(param['transl']).to(self.device)
            torch_param['global_orient'] = torch.tensor(param['global_orient']).to(self.device)
            torch_param['betas'] = torch.tensor(param['betas']).to(self.device)
            torch_param['body_pose'] = torch.tensor(param['body_pose']).to(self.device)
            smpl_output = self.smplx_neutral(return_verts=True, **torch_param)  # generated human body mesh
            joints_cam = smpl_output.joints[:, 0:self.joints_num, :]  # [1, 22, 3]

            # joints to world coordinate
            joints = torch.matmul(self.cam_R, joints_cam.permute(0, 2, 1)).permute(0, 2, 1) + self.cam_t
            joints_list_world.append(joints)
            # smplx params to world (scene) coordinate
            smplx_params_dict = {'global_orient': param['global_orient'],
                                 'transl': param['transl'],
                                 'betas': param['betas'],
                                 'body_pose': param['body_pose'],
                                 }
            smplx_params_dict_scene = update_globalRT_for_smplx(smplx_params_dict, cam2world.detach().cpu().numpy(),
                                                              delta_T=joints_cam[:, 0].detach().cpu().numpy() - smplx_params_dict['transl'])
            smplx_list_world.append(np.concatenate([smplx_params_dict_scene['global_orient'], smplx_params_dict_scene['transl'],
                                                    smplx_params_dict_scene['betas'], smplx_params_dict_scene['body_pose']], axis=-1)[0])  # [79]

            ###### read openpose keypoints
            keypoint_path = os.path.join(self.base_dir, 'keypoints_openpose', self.recording_name, cur_frame_name + '_keypoints.json')
            with open(keypoint_path) as keypoint_file:
                data = json.load(keypoint_file)
            if len(data['people']) == 0:
                cur_keypoints_2d = np.zeros((self.joints_num, 3))
            else:
                cur_keypoints_2d = np.array(data['people'][0]['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3] openpose tolopogy
                cur_keypoints_2d = cur_keypoints_2d[self.openpose_to_smpl]  # [22, 3], smpl topology
            keypoints_2d_list.append(cur_keypoints_2d)

        joints_list_world = torch.cat(joints_list_world, dim=0).detach().cpu().numpy()  # [n_total_frame, 22, 3]
        smplx_list_world = np.asarray(smplx_list_world)
        keypoints_2d_list = np.asarray(keypoints_2d_list)
        joint_mask_list = np.load(joint_mask_path)  # [n_total_frames, 25] for smplx 25 body joints

        ############################### divide sequence into short clips with overlapping window
        seq_idx = 0
        self.frame_name_list = []
        self.joints_clip_world_list = []
        self.smplx_clip_world_list = []
        self.keypoints_2d_list = []
        self.joint_mask_list = []
        while 1:
            start = seq_idx * (self.clip_len - self.clip_overlap_len)
            end = start + self.clip_len
            # print(start, end)
            if end > len(joints_list_world):
                break
            self.frame_name_list.append(frame_name_list[start:end])
            self.joints_clip_world_list.append(joints_list_world[start:end][:, 0:self.joints_num])
            self.smplx_clip_world_list.append(smplx_list_world[start:end])  # todo
            self.keypoints_2d_list.append(keypoints_2d_list[start:end][:, 0:self.joints_num])
            self.joint_mask_list.append(joint_mask_list[start:end][:, 0:self.joints_num])
            seq_idx += 1
        self.n_samples = seq_idx

        print('[INFO] PROX sequence {}: get {} sub clips in total.'.format(self.recording_name, self.n_samples))


    def read_data_egobody(self):
        df = pd.read_csv(os.path.join(self.base_dir, 'egobody_rohm_info.csv'))
        recording_name_list = list(df['recording_name'])
        idx_list = list(df['target_idx'])
        gender_list = list(df['target_gender'])
        view_list = list(df['view'])
        scene_name_list = list(df['scene_name'])
        body_idx_fpv_list = list(df['body_idx_fpv'])

        idx_dict = dict(zip(recording_name_list, idx_list))
        gender_dict = dict(zip(recording_name_list, gender_list))
        view_dict = dict(zip(recording_name_list, view_list))
        scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))

        self.view = view_dict[self.recording_name]
        self.body_idx = idx_dict[self.recording_name]
        self.scene_name = scene_name_dict[self.recording_name]
        self.gender_gt = gender_dict[self.recording_name]

        ######################### read split info
        data_split_info = pd.read_csv(os.path.join(self.base_dir, 'data_splits.csv'))
        train_split_list = list(data_split_info['train'])
        val_split_list = list(data_split_info['val'])
        test_split_list = list(data_split_info['test'])
        ######################### check which split to get fitting gt
        if self.recording_name in train_split_list:
            split = 'train'
        elif self.recording_name in val_split_list:
            split = 'val'
        elif self.recording_name in test_split_list:
            split = 'test'
        else:
            print('Error: {} not in all splits.'.format(self.recording_name))
            exit()

        ######################### see if the target is camera_wearer or interactee to get fitting gt root
        interactee_idx = int(body_idx_fpv_dict[self.recording_name].split(' ')[0])
        if idx_dict[self.recording_name] == interactee_idx:
            self.fitting_gt_root = os.path.join(self.base_dir, 'smplx_interactee_{}'.format(split), self.recording_name,
                                             'body_idx_{}'.format(idx_dict[self.recording_name]))
        else:
            self.fitting_gt_root = os.path.join(self.base_dir, 'smplx_camera_wearer_{}'.format(split), self.recording_name,
                                             'body_idx_{}'.format(idx_dict[self.recording_name]))

        fitting_dir = os.path.join(self.init_root, self.recording_name, 'body_idx_{}'.format(idx_dict[self.recording_name]), 'results')
        joint_mask_path = os.path.join(self.base_dir, 'mask_joint', self.recording_name, self.view, 'mask_joint.npy')
        self.scene_floor_height = egobody_floor_height[self.scene_name]

        ######################### load calibration from sub kinect to main kinect (between color cameras)
        # master: kinect 12, sub_1: kinect 11, sub_2: kinect 13, sub_3, kinect 14, sub_4: kinect 15
        calib_trans_dir = os.path.join(self.base_dir, 'calibrations', self.recording_name)  # extrinsics
        with open(os.path.join(calib_trans_dir, 'cal_trans', 'kinect12_to_world', self.scene_name + '.json'), 'r') as f:
            master2world = np.asarray(json.load(f)['trans'])
        if self.view == 'sub_1':
            trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_11to12_color.json')
        elif self.view == 'sub_2':
            trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_13to12_color.json')
        elif self.view == 'sub_3':
            trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_14to12_color.json')
        elif self.view == 'sub_4':
            trans_subtomain_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect_15to12_color.json')
        if self.view != 'master':
            with open(os.path.join(trans_subtomain_path), 'r') as f:
                trans_subtomain = np.asarray(json.load(f)['trans'])
            cam2world = np.matmul(master2world, trans_subtomain)  # subcamera2world
        else:
            cam2world = master2world
        cam2world = torch.from_numpy(cam2world).float().to(self.device)
        self.cam_R = cam2world[:3, :3].reshape([3, 3])
        self.cam_t = cam2world[:3, 3].reshape([1, 3])

        ##### egobody gt body is in the master kinect camera coord
        master2world = torch.from_numpy(master2world).float().to(self.device)
        cam_master_R = master2world[:3, :3].reshape([3, 3])
        cam_master_t = master2world[:3, 3].reshape([1, 3])

        with open(os.path.join(self.base_dir, 'kinect_cam_params', 'kinect_{}'.format(self.view), 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)

        frame_list = os.listdir(fitting_dir)
        frame_list.sort()
        frame_total = len(frame_list)
        print('[INFO] total frames of current sequence: ', frame_total)
        frame_name_list = []
        joints_list_world = []
        joints_list_world_gt = []
        smplx_list_world = []
        smplx_list_world_gt = []
        keypoints_2d_list = []
        for cur_frame_name in tqdm(frame_list):
            frame_name_list.append(cur_frame_name)
            cur_prox_params_dir = os.path.join(fitting_dir, cur_frame_name, '000.pkl')
            with open(cur_prox_params_dir, 'rb') as f:
                param = pkl.load(f)
            with open(os.path.join(self.fitting_gt_root, 'results', cur_frame_name, '000.pkl'), 'rb') as f:
                param_gt = pkl.load(f)
            ######### for initialzied body
            torch_param = {}
            torch_param['transl'] = torch.tensor(param['transl']).to(self.device)
            torch_param['global_orient'] = torch.tensor(param['global_orient']).to(self.device)
            torch_param['betas'] = torch.tensor(param['betas']).to(self.device)
            torch_param['body_pose'] = torch.tensor(param['body_pose']).to(self.device)
            smpl_output = self.smplx_neutral(return_verts=True, **torch_param)  # generated human body mesh
            joints_cam = smpl_output.joints[:, 0:self.joints_num, :]  # [1, 22, 3]
            # joints to world coordinate
            joints_world = torch.matmul(self.cam_R, joints_cam.permute(0, 2, 1)).permute(0, 2, 1) + self.cam_t
            joints_list_world.append(joints_world)
            # smplx params to world coordinate
            smplx_params_dict = {'global_orient': param['global_orient'],
                                 'transl': param['transl'],
                                 'betas': param['betas'],
                                 'body_pose': param['body_pose'],
                                 }
            cano_smplx_params_dict = update_globalRT_for_smplx(smplx_params_dict, cam2world.detach().cpu().numpy(),
                                                               delta_T=joints_cam[:, 0].detach().cpu().numpy() - smplx_params_dict['transl'])
            smplx_list_world.append(np.concatenate([cano_smplx_params_dict['global_orient'], cano_smplx_params_dict['transl'],
                                                    cano_smplx_params_dict['betas'], cano_smplx_params_dict['body_pose']], axis=-1)[0])  # [79]
            ######### for GT body
            torch_param_gt = {}
            torch_param_gt['transl'] = torch.tensor(param_gt['transl']).to(self.device)
            torch_param_gt['global_orient'] = torch.tensor(param_gt['global_orient']).to(self.device)
            torch_param_gt['betas'] = torch.tensor(param_gt['betas']).to(self.device)
            torch_param_gt['body_pose'] = torch.tensor(param_gt['body_pose']).to(self.device)
            if gender_dict[self.recording_name] == 'male':
                smpl_output_gt = self.smplx_male(return_verts=True, **torch_param_gt)
            elif gender_dict[self.recording_name] == 'female':
                smpl_output_gt = self.smplx_female(return_verts=True, **torch_param_gt)
            joints_cam_gt = smpl_output_gt.joints[:, 0:self.joints_num, :]  # [1, 22, 3]
            # joints to world coordinate
            joints_gt_world = torch.matmul(cam_master_R, joints_cam_gt.permute(0, 2, 1)).permute(0, 2, 1) + cam_master_t
            joints_list_world_gt.append(joints_gt_world)
            # smplx params to world coordinate
            smplx_params_dict_gt = {'global_orient': param_gt['global_orient'],
                                    'transl': param_gt['transl'],
                                    'betas': param_gt['betas'],
                                    'body_pose': param_gt['body_pose'],
                                    }
            cano_smplx_params_dict_gt = update_globalRT_for_smplx(smplx_params_dict_gt, cam2world.detach().cpu().numpy(),
                                                                  delta_T=joints_cam_gt[:, 0].detach().cpu().numpy() - smplx_params_dict_gt['transl'])
            smplx_list_world_gt.append(np.concatenate([cano_smplx_params_dict_gt['global_orient'], cano_smplx_params_dict_gt['transl'],
                                                       cano_smplx_params_dict_gt['betas'], cano_smplx_params_dict_gt['body_pose']], axis=-1)[0])  # [79]

            ########### read openpose keypoints
            keypoint_path = os.path.join(self.base_dir, 'keypoints_cleaned', self.recording_name, self.view, cur_frame_name + '_keypoints.json')
            with open(keypoint_path) as keypoint_file:
                data = json.load(keypoint_file)
            if len(data['people']) == 0:
                cur_keypoints_2d = np.zeros((self.joints_num, 3))
            else:
                cur_keypoints_2d = np.array(data['people'][self.body_idx]['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3] openpose tolopogy
                cur_keypoints_2d = cur_keypoints_2d[self.openpose_to_smpl]  # [22, 3], smpl topology
            keypoints_2d_list.append(cur_keypoints_2d)

        joints_list_world = torch.cat(joints_list_world, dim=0).detach().cpu().numpy()  # [n_total_frame, 22, 3]
        smplx_list_world = np.asarray(smplx_list_world)
        joints_list_world_gt = torch.cat(joints_list_world_gt, dim=0).detach().cpu().numpy()  # [n_total_frame, 22, 3]
        smplx_list_world_gt = np.asarray(smplx_list_world_gt)
        keypoints_2d_list = np.asarray(keypoints_2d_list)
        joint_mask_list = np.load(joint_mask_path)  # [n_total_frames, 25] for smplx 25 body joints

        ############################### divide sequence into short clips with overlapping window
        seq_idx = 0
        self.frame_name_list = []
        self.joints_clip_world_list = []
        self.joints_clip_world_list_gt = []
        self.smplx_clip_world_list = []
        self.smplx_clip_world_list_gt = []
        self.keypoints_2d_list = []
        self.joint_mask_list = []
        while 1:
            start = seq_idx * (self.clip_len - self.clip_overlap_len)
            end = start + self.clip_len
            # print(start, end)
            if end > len(joints_list_world):
                break
            self.frame_name_list.append(frame_name_list[start:end])
            self.joints_clip_world_list.append(joints_list_world[start:end][:, 0:self.joints_num])
            self.smplx_clip_world_list.append(smplx_list_world[start:end])
            self.joints_clip_world_list_gt.append(joints_list_world_gt[start:end][:, 0:self.joints_num])
            self.smplx_clip_world_list_gt.append(smplx_list_world_gt[start:end])  # todo
            self.keypoints_2d_list.append(keypoints_2d_list[start:end][:, 0:self.joints_num])
            self.joint_mask_list.append(joint_mask_list[start:end][:, 0:self.joints_num])
            seq_idx += 1
        self.n_samples = seq_idx

        print('[INFO] EgoBody sequence {}: get {} sub clips in total.'.format(self.recording_name, self.n_samples))


    def create_body_repr(self):
        for i in tqdm(range(0, self.n_samples, 1)):
            source_data_joints = self.joints_clip_world_list[i][:, 0:self.joints_num, :]  # [T, 22, 3]
            source_data_smplx = self.smplx_clip_world_list[i]  # [T, 79]

            smplx_params_dict = {'global_orient': source_data_smplx[:, 0:3],
                                 'transl': source_data_smplx[:, 3:6],
                                 'betas': source_data_smplx[:, 6:16],
                                 'body_pose': source_data_smplx[:, 16:(16 + 63)],
                                 }  # for the clip

            ############## canonicalize for input initialized sequence
            cano_fn = cano_seq_smplx if self.dataset=='prox' else cano_seq_smplx_egobody
            # cano_positions/cano_smplx_params_dict: joints/smplx params in the canonicalized coord system
            # transf_matrix: transformation matrix from scene(world) coord to canonicalized coord system
            cano_positions, cano_smplx_params_dict, transf_matrix = cano_fn(positions=source_data_joints,
                                                                            smplx_params_dict=smplx_params_dict,
                                                                            preset_floor_height=self.scene_floor_height if self.use_scene_floor_height else None,
                                                                            return_transf_mat=True,
                                                                            smpl_model=self.smplx_neutral,
                                                                            device=self.device)
            ############## create motion representation
            repr_dict = get_repr_smplx(positions=cano_positions, smplx_params_dict=cano_smplx_params_dict,
                                       feet_vel_thre=5e-5)  # a dict of reprs

            #######
            self.cano_smplx_params_dict_list.append(cano_smplx_params_dict)
            self.transf_matrix_list.append(transf_matrix)
            self.cano_joints_input_list.append(cano_positions)
            for repr_name in REPR_LIST:
                self.repr_list_input_dict[repr_name].append(repr_dict[repr_name])

        #######################################  get mean/std from training dataset
        save_dir = self.logdir
        for repr_name in REPR_LIST:
            self.repr_list_input_dict[repr_name] = np.asarray(self.repr_list_input_dict[repr_name])  # each item: [N, T-1, d]
        with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'rb') as f:
            self.Mean_dict = pkl.load(f)
        with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'rb') as f:
            self.Std_dict = pkl.load(f)
        self.Mean = np.concatenate([self.Mean_dict[key] for key in self.Mean_dict.keys()], axis=-1)
        self.Std = np.concatenate([self.Std_dict[key] for key in self.Std_dict.keys()], axis=-1)


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        repr_dict = {}
        for repr_name in REPR_LIST:
            repr_dict[repr_name] = self.repr_list_input_dict[repr_name][index]  # [clip_len, d]

        item_dict = {}
        item_dict['motion_repr_noisy'] = np.concatenate([repr_dict[key] for key in REPR_LIST], axis=-1)  # [clip_len-1, body_feat_dim]
        item_dict['motion_repr_noisy'] = ((item_dict['motion_repr_noisy'] - self.Mean) / self.Std).astype(np.float32)
        item_dict['noisy_joints'] = self.cano_joints_input_list[index].astype(np.float32)  # in canonicalized coord
        item_dict['noisy_joints_scene_coord'] = self.joints_clip_world_list[index].astype(np.float32)  # [clip_len, 22, 3] in scene coord
        if self.dataset == 'egobody':
            item_dict['gt_joints_scene_coord'] = self.joints_clip_world_list_gt[index].astype(np.float32)  # [clip_len, 22, 3] in scene coord
        item_dict['transf_matrix'] = self.transf_matrix_list[index].astype(np.float32)
        item_dict['cano_smplx_params_dict'] = self.cano_smplx_params_dict_list[index]
        for key in item_dict['cano_smplx_params_dict']:
            item_dict['cano_smplx_params_dict'][key] = item_dict['cano_smplx_params_dict'][key].astype(np.float32)
        item_dict['frame_name'] = self.frame_name_list[index]
        item_dict['focal_length'] = np.array([self.color_cam['f'][0], self.color_cam['f'][1]]).astype(np.float32)  # [2]
        item_dict['camera_center'] = np.array([self.color_cam['c'][0], self.color_cam['c'][1]]).astype(np.float32)  # [2]

        keypoints_2d_orig = self.keypoints_2d_list[index]
        if self.dataset == 'prox':
            ### fist un-flip openpose joints
            keypoints_temp_0 = np.zeros(self.keypoints_2d_list[index].shape)  # [clip_len, 22, 3]
            keypoints_temp_0[:, :, 0] = 1920 - 1 - keypoints_2d_orig[:, :, 0]
            keypoints_temp_0[:, :, 1:] = keypoints_2d_orig[:, :, 1:]
            ### then undistort
            distorted_points = cv2.undistortPoints(
                src=(keypoints_temp_0[:, :, :2]).copy().reshape(-1, 2),
                cameraMatrix=np.asarray(self.color_cam['camera_mtx']),
                distCoeffs=np.asarray(self.color_cam['k']),
                P=np.asarray(self.color_cam['camera_mtx'])).squeeze()  # [clip_len*22, 2]
            keypoints_temp_0[:, :, :2] = distorted_points.reshape(keypoints_temp_0.shape[0], -1, 2)
            ### then flip-back openpose joints
            keypoints_temp_1 = np.zeros(keypoints_temp_0.shape)
            keypoints_temp_1[:, :, 0] = 1920 - 1 - keypoints_temp_0[:, :, 0]
            keypoints_temp_1[:, :, 1:] = keypoints_temp_0[:, :, 1:]
            item_dict['keypoints_2d'] = keypoints_temp_1  # [clip_len, 22, 3]  openpose joints in smpl topology with conf score
        elif self.dataset == 'egobody':
            item_dict['keypoints_2d'] = keypoints_2d_orig  # [clip_len, 22, 3]  openpose joints in smpl topology with conf score

        mask_keypoint2d_conf = self.keypoints_2d_list[index][:, :, -1] > 0.2  # [clip_len, 22] 1 for visible
        mask_joint_depth = self.joint_mask_list[index]  # [clip_len, 22, 3]
        mask_joint_vis = mask_keypoint2d_conf * mask_joint_depth  # [clip_len, 22]
        item_dict['mask_joint_vis'] = mask_joint_vis.astype(np.float32)   # [clip_len, 22]  1 for visible

        clip_len = len(mask_joint_vis)
        mask_vec_vis = {}
        for key in REPR_LIST:
            if key in ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel', 'root_height',
                       'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel', 'smplx_betas']:
                mask_vec_vis[key] = np.ones((clip_len, REPR_DIM_DICT[key]))
            elif key in ['local_positions', 'local_vel']:
                mask_vec_vis[key] = mask_joint_vis[:, 0:].repeat(3, axis=1)
            elif key == 'smplx_body_pose_6d':
                mask_vec_vis[key] = mask_joint_vis[:, 1:].repeat(6, axis=1)
            elif key == 'foot_contact':
                mask_vec_vis[key] = np.zeros((clip_len, 4))
                left_foot_visible = (mask_joint_vis[:, 7] == 1) * (mask_joint_vis[:, 10] == 1)  # [T]
                right_foot_visible = (mask_joint_vis[:, 8] == 1) * (mask_joint_vis[:, 11] == 1)  # [T]
                mask_vec_vis[key][left_foot_visible, 0:2] = 1.0
                mask_vec_vis[key][right_foot_visible, 2:] = 1.0
        mask_vec_vis = np.concatenate([mask_vec_vis[key] for key in mask_vec_vis], axis=-1)  # [T, body_feat_dim]
        item_dict['mask_vec_vis'] = mask_vec_vis.astype(np.float32)  # [clip_len, body_feat_dim]

        ####### set up traj network condition input
        if self.task == 'traj':
            if not self.repr_abs_only:
                noisy_traj = item_dict['motion_repr_noisy'][:, 0:self.traj_feat_dim]
            else:
                temp = item_dict['motion_repr_noisy']
                noisy_traj = np.concatenate([temp[..., [0]], temp[..., 2:4], temp[..., [6]],
                                             temp[..., 7:13], temp[..., 16:19]],
                                            axis=-1)  # [clip_len-1, 13]
            item_dict['cond'] = noisy_traj
            item_dict['control_cond'] = item_dict['motion_repr_noisy'][:, -self.pose_feat_dim:]

        return item_dict
