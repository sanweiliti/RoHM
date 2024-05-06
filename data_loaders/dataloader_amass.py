from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_loaders.motion_representation import *
import pickle as pkl
from utils.other_utils import REPR_LIST, REPR_DIM_DICT



class DataloaderAMASS(data.Dataset):
    def __init__(self,
                 preprocessed_amass_root='',
                 body_model_path='',
                 amass_datasets=None,
                 split='train',
                 spacing=1,
                 repr_abs_only=False,
                 input_noise=False,
                 sep_noise=False,
                 noise_std_joint=0.0,
                 noise_std_smplx_global_rot=0.0,
                 noise_std_smplx_body_rot=0.0,
                 noise_std_smplx_trans=0.0,
                 noise_std_smplx_betas=0.0,
                 load_noise=False,
                 loaded_smplx_noise_dict=None,
                 task='traj',
                 clip_len=150, joints_num=22,
                 logdir=None, device='cpu'):
        self.preprocessed_amass_root = preprocessed_amass_root
        self.split = split
        self.clip_len = clip_len
        self.logdir = logdir
        self.device = device
        self.spacing = spacing
        self.smplx_neutral = smplx.create(model_path=body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)

        self.joints_num = joints_num
        self.head_joint_idx = [15]  # smpl joint 15: chin actually
        self.torso_joint_idx = [12, 9, 6, 3]  # neck, spine

        self.task = task
        if self.task not in ['traj', 'pose']:
            print('[ERROR] args.traj should be in [traj, pose]!')
            exit()

        ########################################## configs about how to add noise
        self.input_noise = input_noise
        self.sep_noise = sep_noise  # add different noise to joint-based repr, and smplx-based repr, separately
        self.noise_std_joint = noise_std_joint  # set if sep_noise=True
        self.noise_std_params_dict = {'global_orient': noise_std_smplx_global_rot,
                                      'transl': noise_std_smplx_trans,
                                      'body_pose': noise_std_smplx_body_rot,
                                      'betas': noise_std_smplx_betas,}
        self.load_noise = load_noise  # load preset noise (for reproducibility)
        self.loaded_smplx_noise_dict = loaded_smplx_noise_dict  # loaded preset noise

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

        self.repr_list_dict = {}
        for repr_name in REPR_LIST:
            self.repr_list_dict[repr_name] = []
        self.smplx_params_list_dict = {}  # each item: params of all samples
        for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
            self.smplx_params_list_dict[param_name] = []

        self.joints_clean_list = []
        if self.input_noise and (not self.sep_noise):
            self.joints_noisy_list = []
            self.repr_list_dict_noisy = {}
            for repr_name in REPR_LIST:
                self.repr_list_dict_noisy[repr_name] = []

        self.joints_clip_list = []
        self.smplx_clip_list = []

        ######################################## read data and compute the motion reprentations
        self.read_data(amass_datasets)
        self.create_body_repr()


    def divide_clip(self, dataset_name='HumanEva'):
        preprocessed_amass_joints_dir = os.path.join(self.preprocessed_amass_root, 'pose_data_fps_30')
        preprocessed_amass_smpl_dir = os.path.join(self.preprocessed_amass_root, 'smpl_data_fps_30')
        seqs_path = glob.glob(os.path.join(preprocessed_amass_joints_dir, dataset_name, '*/*.npy'))  # name list of all npz sequence files in current dataset
        seqs_path = sorted(seqs_path)
        # print('reading sequences in %s...' % (dataset_name))
        for path in seqs_path:
            seq_name = path.split('/')[-2]
            npy_name = path.split('/')[-1]
            path_joints = os.path.join(preprocessed_amass_joints_dir, dataset_name, seq_name, npy_name)
            path_smplx = os.path.join(preprocessed_amass_smpl_dir, dataset_name, seq_name, npy_name)
            seq_joints = np.load(path_joints)  # [seq_len, 25, 3]
            seq_smplx = np.load(path_smplx) # [seq_len, 178]
            if self.split == 'test':
                seq_joints = seq_joints[1:-1]
                seq_smplx = seq_smplx[1:-1]
            N = len(seq_joints)  # total frame number of the current sequence
            # divide long sequences into sub clips
            if N >= self.clip_len:
                num_valid_clip = int(N / self.clip_len)
                for i in range(num_valid_clip):
                    joints_clip = seq_joints[(self.clip_len * i):self.clip_len * (i + 1)]  # [clip_len, 25, 3]
                    smplx_clip = seq_smplx[(self.clip_len * i):self.clip_len * (i + 1)]  # [clip_len, ]
                    self.joints_clip_list.append(joints_clip)
                    self.smplx_clip_list.append(smplx_clip)
            else:
                continue

    def read_data(self, amass_datasets):
        for dataset_name in tqdm(amass_datasets):
            self.divide_clip(dataset_name)
        self.n_samples = len(self.joints_clip_list)
        print('[INFO] {} set: get {} sub clips in total.'.format(self.split, self.n_samples))


    def create_body_repr(self):
        smplx_noise_dict = {}
        for i in tqdm(range(0, self.n_samples, self.spacing)):
            source_data_joints = self.joints_clip_list[i][:, 0:self.joints_num, :]  # [T, 22, 3]
            source_data_smplx = self.smplx_clip_list[i]  # [T, 178]
            smplx_params_dict = {'global_orient': source_data_smplx[:, 0:3],
                                 'transl': source_data_smplx[:, 3:6],
                                 'betas': source_data_smplx[:, 6:16],
                                 'body_pose': source_data_smplx[:, 16:(16+63)],
                                 }  # for the clip

            ######################################## canonicalize for GT sequence
            cano_positions, cano_smplx_params_dict = cano_seq_smplx(positions=source_data_joints,
                                                                    smplx_params_dict=smplx_params_dict,
                                                                    smpl_model=self.smplx_neutral, device=self.device)

            ######################################## add noise to smplx params
            if self.input_noise and (not self.sep_noise):
                cano_smplx_params_dict_noisy = {}
                for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
                    if param_name == 'transl' or param_name == 'betas':
                        if self.load_noise:
                            noise_1 = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smplx_params_dict[param_name].shape)
                        cano_smplx_params_dict_noisy[param_name] = cano_smplx_params_dict[param_name] + noise_1
                        if param_name not in smplx_noise_dict.keys():
                            smplx_noise_dict[param_name] = []
                        smplx_noise_dict[param_name].append(noise_1)
                    elif param_name == 'global_orient':
                        global_orient_mat = R.from_rotvec(cano_smplx_params_dict['global_orient'])  # [145, 3, 3]
                        global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
                        if self.load_noise:
                            noise_global_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                        global_orient_angle_noisy = global_orient_angle + noise_global_rot
                        cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
                        if param_name not in smplx_noise_dict.keys():
                            smplx_noise_dict[param_name] = []
                        smplx_noise_dict[param_name].append(noise_global_rot)  #  [145, 3] in euler angle
                    elif param_name == 'body_pose':
                        body_pose_mat = R.from_rotvec(cano_smplx_params_dict['body_pose'].reshape(-1, 3))
                        body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
                        if self.load_noise:
                            noise_body_pose_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
                        else:
                            noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                        body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                        cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
                        if param_name not in smplx_noise_dict.keys():
                            smplx_noise_dict[param_name] = []
                        smplx_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, 21, 3))  # [145, 21, 3]  in euler angle

                ### using FK to obtain noisy joint positions from noisy smplx params
                smplx_params_dict_noisy_torch = {}
                for key in cano_smplx_params_dict_noisy.keys():
                    smplx_params_dict_noisy_torch[key] = torch.FloatTensor(cano_smplx_params_dict_noisy[key]).to(self.device)
                bs = smplx_params_dict_noisy_torch['transl'].shape[0]
                # we do not consider face/hand details in RoHM
                smplx_params_dict_noisy_torch['jaw_pose'] = torch.zeros(bs, 3).to(self.device)
                smplx_params_dict_noisy_torch['leye_pose'] = torch.zeros(bs, 3).to(self.device)
                smplx_params_dict_noisy_torch['reye_pose'] = torch.zeros(bs, 3).to(self.device)
                smplx_params_dict_noisy_torch['left_hand_pose'] = torch.zeros(bs, 45).to(self.device)
                smplx_params_dict_noisy_torch['right_hand_pose'] = torch.zeros(bs, 45).to(self.device)
                smplx_params_dict_noisy_torch['expression'] = torch.zeros(bs, 10).to(self.device)
                cano_positions_noisy = self.smplx_neutral(**smplx_params_dict_noisy_torch).joints[:, 0:22].detach().cpu().numpy()  # [clip_len, 22, 3]

            ######################################## create motion representation
            repr_dict = get_repr_smplx(positions=cano_positions,
                                       smplx_params_dict=cano_smplx_params_dict,
                                       feet_vel_thre=5e-5)  # a dict of reprs
            if self.input_noise and (not self.sep_noise):
                repr_dict_noisy = get_repr_smplx(positions=cano_positions_noisy,
                                                 smplx_params_dict=cano_smplx_params_dict_noisy,
                                                 feet_vel_thre=5e-5)  # a dict of reprs

            ############### clean data repr gt
            self.joints_clean_list.append(cano_positions)
            for repr_name in REPR_LIST:
                self.repr_list_dict[repr_name].append(repr_dict[repr_name])
            for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                self.smplx_params_list_dict[param_name].append(cano_smplx_params_dict[param_name])

            if self.input_noise and (not self.sep_noise):
                self.joints_noisy_list.append(cano_positions_noisy)
                for repr_name in REPR_LIST:
                    self.repr_list_dict_noisy[repr_name].append(repr_dict_noisy[repr_name])


            ######### FOR DEBUG: rec_ric_data should be same as cano_positions
            # repr_dict_torch = {}
            # for key in repr_dict.keys():
            #     repr_dict_torch[key] = torch.from_numpy(repr_dict[key]).unsqueeze(0).float().to(self.device)
            # rec_ric_data_clean = recover_from_repr_smpl(repr_dict_torch,
            #                                             recover_mode='smplx_params', smplx_model=self.smplx_neutral)  # [1, T-1, 22, 3]
            # rec_ric_data_clean = rec_ric_data_clean.detach().cpu().numpy()[0]  # [T-1, 22, 3]

        # ####################################### save smplx param noise
        # import pickle
        # for param_name in smplx_noise_dict.keys():
        #     smplx_noise_dict[param_name] = np.asarray(smplx_noise_dict[param_name])
        # pkl_path = 'eval_noise_smplx/smplx_noise_level_9.pkl'
        # with open(pkl_path, 'wb') as result_file:
        #     pickle.dump(smplx_noise_dict, result_file, protocol=2)
        # print('current smplx noise saved to{}.'.format(pkl_path))

        #######################################  get mean/std for dataset
        save_dir = self.logdir
        for repr_name in REPR_LIST:
            self.repr_list_dict[repr_name] = np.asarray(self.repr_list_dict[repr_name])  # each item: [N, T-1, d]
        if self.split == 'train':
            self.Mean_dict = {}
            self.Std_dict = {}
            for repr_name in REPR_LIST:
                self.Mean_dict[repr_name] = self.repr_list_dict[repr_name].reshape(-1, REPR_DIM_DICT[repr_name]).mean(axis=0).astype(np.float32)
                if repr_name == 'foot_contact':
                    self.Mean_dict[repr_name][...] = 0.0
                self.Std_dict[repr_name] = self.repr_list_dict[repr_name].reshape(-1, REPR_DIM_DICT[repr_name]).std(axis=0).astype(np.float32)
                # do not normalize for smplx beta (already in a normal distribution) and foot contact labels (0/1 label)
                if repr_name != 'smplx_betas' and repr_name != 'foot_contact':
                    self.Std_dict[repr_name][...] = self.Std_dict[repr_name].mean() / 1.0
                elif repr_name == 'foot_contact':
                    self.Std_dict[repr_name][...] = 1.0
            ######## save mean/std stats for the training data
            os.makedirs(save_dir) if not os.path.exists(save_dir) else None
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'wb') as result_file:
                pkl.dump(self.Mean_dict, result_file, protocol=2)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'wb') as result_file:
                pkl.dump(self.Std_dict, result_file, protocol=2)

        elif self.split == 'test':
            ######## load mean/std stats from the training data
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'rb') as f:
                self.Mean_dict = pkl.load(f)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'rb') as f:
                self.Std_dict = pkl.load(f)

        self.Mean = np.concatenate([self.Mean_dict[key] for key in self.Mean_dict.keys()], axis=-1)
        self.Std = np.concatenate([self.Std_dict[key] for key in self.Std_dict.keys()], axis=-1)


    def __len__(self):
        return self.n_samples // self.spacing

    def __getitem__(self, index):
        positions_clean = self.joints_clean_list[index]
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = self.repr_list_dict[repr_name][index]  # [clip_len, d]

        ####################################### add noise
        if self.input_noise:
            if self.sep_noise:
                ## add different noise to smplx params and joint positions separately
                smplx_params_dict_clean = {}
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    smplx_params_dict_clean[param_name] = self.smplx_params_list_dict[param_name][index]  # [clip_len, d]
                #### add noise to smpl params
                smplx_params_dict_noisy = {}
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    noise = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name],
                                             size=smplx_params_dict_clean[param_name].shape)
                    smplx_params_dict_noisy[param_name] = smplx_params_dict_clean[param_name] + noise
                ### add noise to joint positions
                noise_0 = np.random.normal(loc=0.0, scale=self.noise_std_joint, size=positions_clean.shape)
                positions_noisy = positions_clean + noise_0  # [clip_len, 22, 3]
                positions_noisy = positions_noisy.astype(np.float32)
                ######## creat noisy repr
                repr_dict_noisy = get_repr_smplx(positions=positions_noisy, smplx_params_dict=smplx_params_dict_noisy,)
            else:
                ## add noise to smplx params, and noisy joints obtained by FK (our setup)
                positions_noisy = self.joints_noisy_list[index]
                repr_dict_noisy = {}
                for repr_name in REPR_LIST:
                    repr_dict_noisy[repr_name] = self.repr_list_dict_noisy[repr_name][index]  # [clip_len, d]

        ####################################### get data items
        item_dict = {}
        item_dict['motion_repr_clean'] = np.concatenate([repr_dict_clean[key] for key in REPR_LIST], axis=-1) # [clip_len-1, 263]
        if self.input_noise:
            item_dict['noisy_joints'] = positions_noisy
            item_dict['motion_repr_noisy'] = np.concatenate([repr_dict_noisy[key] for key in REPR_LIST], axis=-1) # [clip_len-1, 263]
            if self.task == 'pose':  # PoseNet conditioned on clean traj input
                item_dict['motion_repr_noisy'][:, 0:self.traj_feat_dim] = item_dict['motion_repr_clean'][:, 0:self.traj_feat_dim]
        else:
            item_dict['motion_repr_noisy'] = item_dict['motion_repr_clean'].copy()

        item_dict['motion_repr_clean'] = ((item_dict['motion_repr_clean'] - self.Mean) / self.Std).astype(np.float32)
        item_dict['motion_repr_noisy'] = ((item_dict['motion_repr_noisy'] - self.Mean) / self.Std).astype(np.float32)

        if self.task == 'traj':
            if not self.repr_abs_only:
                noisy_traj = item_dict['motion_repr_noisy'][:, 0:self.traj_feat_dim]
            else:
                # if repr_abs_only=False, exclude traj velocities
                temp = item_dict['motion_repr_noisy']
                noisy_traj = np.concatenate([temp[..., [0]], temp[..., 2:4], temp[..., [6]], temp[..., 7:13], temp[..., 16:19]], axis=-1)  # [144, 13]
            item_dict['cond'] = noisy_traj  # condition of TrajNet: noisy trajectory
            item_dict['control_cond'] = item_dict['motion_repr_clean'][:, -self.pose_feat_dim:]  # PoseControl signal: clean local pose features

        return item_dict
