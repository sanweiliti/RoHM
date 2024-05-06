# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/EricGuo5513/HumanML3D

from data_loaders.common.quaternion import *
from utils.konia_transform import *

import math
from utils.other_utils import *

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
# joints_num = 22
head_joint_indx = 15



def foot_detect(positions, thres, up_axis='y'):
    if up_axis == 'y':
        up_axis_dim = 1
    elif up_axis == 'z':
        up_axis_dim = 2
    # velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.18, 0.15])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1,fid_l,up_axis_dim]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
    # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1,fid_r,up_axis_dim]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
    # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r


def cano_seq_smplx(positions, smplx_params_dict, preset_floor_height=None, return_transf_mat=False, smpl_model=None, device='cpu'):
    '''
    Perform canonicalization to the original motion sequence, such that:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    Use for AMASS and PROX (coordinate system z axis up)
    input:
        - positions: original joint positions (z axis up)
        - smplx_params_dict: original smplx params
        - preset_floor_height: if not None, the preset floor height
        - return_transf_mat: if True, also return the transf matrix for canonicalization
    Output:
        - cano_positions: canonicalized joint positions
        - cano_smplx_params_dict: canonicalized smplx params
        - transf_matrix (if return_transf_mat): the transf matrix for canonicalization
    '''
    ##### positions: z axis up
    cano_positions = positions.copy()
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

    ######################## Put on Floor
    if preset_floor_height:
        floor_height = preset_floor_height
    else:
        floor_height = cano_positions.min(axis=0).min(axis=0)[2]
    cano_positions[:, :, 2] -= floor_height  # z: up-axis, foot on ground
    # print(floor_height)

    ######################## transl such that XY for frame 0 is at origin
    root_pos_init = cano_positions[0]  # [22, 3]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    cano_positions = cano_positions - root_pose_init_xy

    ######################## transfrom such that frame 0 faces y+ axis
    joints_frame0 = cano_positions[0] # [N, 3] joints of first frame
    across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
    across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
    x_axis = across1 + across2
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0, 0, 1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    transf_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)  # [3, 3]
    cano_positions = np.matmul(cano_positions, transf_rotmat)  # [T(/bs), 22, 3]

    ######################## canonicalization transf matrix for smpl params
    transf_matrix_1 = np.array([[1, 0, 0, -root_pose_init_xy[0]],
                                [0, 1, 0, -root_pose_init_xy[1]],
                                [0, 0, 1, -floor_height],
                                [0, 0, 0, 1]])
    transf_matrix_2 = np.zeros([4, 4])
    transf_matrix_2[0:3, 0:3] = transf_rotmat.T
    transf_matrix_2[-1, -1] = 1
    transf_matrix = np.matmul(transf_matrix_2, transf_matrix_1)
    cano_smplx_params_dict = update_globalRT_for_smplx(smplx_params_dict, transf_matrix,
                                                      delta_T=positions[:, 0]-smplx_params_dict['transl'])

    if not return_transf_mat:
        return cano_positions, cano_smplx_params_dict
    else:
        return cano_positions, cano_smplx_params_dict, transf_matrix


def cano_seq_smplx_egobody(positions, smplx_params_dict, preset_floor_height=None, return_transf_mat=False, smpl_model=None, device='cpu'):
    '''
    Perform canonicalization to the input motion sequence, similar as cano_seq_smplx,
    but the original input here is y-axis up, use for EgoBody (EgoBody scene coordinate system is y axis up)
    Output:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    '''
    ##### positions: y axis up
    cano_positions = positions.copy()
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

    ######################## Put on Floor
    if preset_floor_height:
        floor_height = preset_floor_height
    else:
        floor_height = cano_positions.min(axis=0).min(axis=0)[1]
    cano_positions[:, :, 1] -= floor_height
    # print(floor_height)

    ######################## transl to XZ at origin
    root_pos_init = cano_positions[0]  # [22, 3]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    cano_positions = cano_positions - root_pose_init_xz  # such that the x/z of first frame is 0

    ######################## transfrom to frame 0 face z+ axis
    joints_frame0 = cano_positions[0] # [N, 3] joints of first frame
    across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
    across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
    x_axis = across1 + across2
    # x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
    x_axis[1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    transf_rotmat = np.stack([x_axis, z_axis, y_axis], axis=1)  # [3, 3]
    transf_rotmat = -transf_rotmat # to make transf_rotmat a valid rotation matrix
    cano_positions = np.matmul(cano_positions, transf_rotmat)  # after this y axis down

    ########################## rotate such to make z axis up
    # rotate around x axis for -90 degrees
    trans_rot_x = np.array([[1, 0, 0],
                            [0, math.cos(-math.pi/2), -math.sin(-math.pi/2)],
                            [0, math.sin(-math.pi/2), math.cos(-math.pi/2)]])
    # rotate around z axis for 180 degrees
    trans_rot_z = np.array([[math.cos(math.pi), -math.sin(math.pi), 0],
                            [math.sin(math.pi), math.cos(math.pi), 0],
                            [0, 0, 1]])
    add_trans = np.matmul(trans_rot_z, trans_rot_x)
    cano_positions = np.matmul(cano_positions, add_trans.T) # after this z axis up

    ######################## canonicalization for smplx params
    transf_matrix_1 = np.array([[1, 0, 0, -root_pose_init_xz[0]],
                                [0, 1, 0, -floor_height],
                                [0, 0, 1, -root_pose_init_xz[2]],
                                [0, 0, 0, 1]])
    transf_matrix_2 = np.identity(4)
    transf_matrix_2[0:3, 0:3] = transf_rotmat.T
    transf_matrix_3 = np.identity(4)
    transf_matrix_3[0:3, 0:3] = add_trans
    transf_matrix = np.matmul(transf_matrix_3, np.matmul(transf_matrix_2, transf_matrix_1))
    cano_smplx_params_dict = update_globalRT_for_smplx(smplx_params_dict, transf_matrix,
                                                      # smplx_model=smpl_model, device=device,
                                                       delta_T=positions[:, 0]-smplx_params_dict['transl'],)

    if not return_transf_mat:
        return cano_positions, cano_smplx_params_dict
    else:
        return cano_positions, cano_smplx_params_dict, transf_matrix


def get_repr_smplx(positions, smplx_params_dict, feet_vel_thre=5e-5):
    '''
    calculate the motion representation for input sequence
    input:
        - positions: input joint positions
        - smplx_params_dict: input smplx params
        - feet_vel_thre: velocity threshold for foot contact label
    Output:
        - data_dict: our motion representation (both traj and local pose)
    '''
    global_positions = positions.copy()

    """ Get Foot Contacts """
    feet_l, feet_r = foot_detect(positions, feet_vel_thre, up_axis='z')  # feet_l/feet_r: [seq_len, 2]

    ##################### joint-based repr #####################
    '''Get Forward Direction'''
    l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
    across1 = positions[:, r_hip] - positions[:, l_hip]
    across2 = positions[:, sdr_r] - positions[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]
    forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    '''Get Root Rotation and rotation velocity'''
    target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
    root_rot_quat = qbetween_np(forward, target)
    ### several frames have nan values
    if np.isnan(root_rot_quat).sum() > 0:
        frame_idx = np.where(np.isnan(root_rot_quat) == True)[0][0]
        root_rot_quat[frame_idx] = root_rot_quat[frame_idx - 1]
    root_rot_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
    root_rot_quat_vel = qmul_np(root_rot_quat[1:], qinv_np(root_rot_quat[:-1]))

    '''abs root traj '''
    root_l_pos = positions[:, 0]  # [seq_len, 3]
    root_height = positions[:, 0, 2:3] # [seq_len]
    '''Get Root linear velocity'''
    root_l_vel = (positions[1:, 0] - positions[:-1, 0]).copy()
    root_l_vel = qrot_np(root_rot_quat[1:], root_l_vel)

    '''Get Root rotation angle and rot velocity angle'''
    root_rot_angle = np.arctan2(root_rot_quat[:, 3:4], root_rot_quat[:, 0:1])  # rotation angle, is half of the actual angle...
    root_rot_angle_vel = np.arctan2(root_rot_quat_vel[:, 3:4], root_rot_quat_vel[:, 0:1])  # rotation angle velocity

    '''local joint positions'''
    local_positions = positions.copy()  # [seq_len, 22, 3]
    local_positions[..., 0] -= local_positions[:, 0:1, 0]
    local_positions[..., 1] -= local_positions[:, 0:1, 1]
    '''for each frame, local pose face y+'''
    local_positions = qrot_np(np.repeat(root_rot_quat[:, None], local_positions.shape[1], axis=1), local_positions)

    '''local joint velocity'''
    local_vel = qrot_np(np.repeat(root_rot_quat[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])

    ##################### smplx-based repr #####################
    '''smpl global_orient matrix'''
    smplx_rot_aa = smplx_params_dict['global_orient']   # [seq_len, 3]
    smplx_rot_mat = R.from_rotvec(smplx_rot_aa).as_matrix()  # [seq_len, 3, 3]
    smplx_rot_6d = smplx_rot_mat[..., :-1].reshape(-1, 6)  # [seq_len, 6]

    '''smpl global_orient velocity'''
    dRdt = smplx_rot_mat[1:] - smplx_rot_mat[0:-1]  # [seq_len-1, 3, 3]
    smplx_rot_vel = estimate_angular_velocity_np(smplx_rot_mat[0:-1], dRdt)

    '''smpl transl and velocity'''
    smplx_trans = smplx_params_dict['transl']  # [seq_len, 3]
    smplx_trans_vel = (smplx_trans[1:] - smplx_trans[:-1]).copy()

    '''smpl body pose'''
    smplx_body_pose_aa = smplx_params_dict['body_pose']  # [seq_len, 63]
    smplx_body_pose_mat = R.from_rotvec(smplx_body_pose_aa.reshape(-1, 3)).as_matrix().reshape(len(smplx_body_pose_aa), -1, 3, 3)  # [seq_len*21, 3, 3]
    smplx_body_pose_6d = smplx_body_pose_mat[..., :-1].reshape([smplx_body_pose_mat.shape[0], -1, 6])

    '''smpl shape'''
    smplx_betas = smplx_params_dict['betas']  # [seq_len, 10]

    ################### final full body repr #####################
    data_dict = {'root_rot_angle': root_rot_angle[0:-1],
                 'root_rot_angle_vel': root_rot_angle_vel,
                 'root_l_pos': root_l_pos[0:-1, [0, 1]],
                 'root_l_vel': root_l_vel[:, [0, 1]],
                 'root_height': root_height[:-1],
                 'smplx_rot_6d': smplx_rot_6d[0:-1],
                 'smplx_rot_vel': smplx_rot_vel,
                 'smplx_trans': smplx_trans[0:-1],
                 'smplx_trans_vel': smplx_trans_vel,
                 'local_positions': local_positions[0:-1].reshape(len(smplx_trans_vel), -1),
                 'smplx_body_pose_6d': smplx_body_pose_6d[0:-1].reshape(len(smplx_trans_vel), -1),
                 'local_vel': local_vel.reshape(len(smplx_trans_vel), -1),
                 'smplx_betas': smplx_betas[0:-1],
                 'foot_contact': np.concatenate([feet_l, feet_r], axis=-1),
                 }
    return data_dict


def recover_root_rot_pos(data, root_traj_repr='abs', up_axis='y'):
    '''
    Recover joint-based trajectory (root linear position and rotation) from full motion representation
    input:
        - data: motion representation
        - root_traj_repr: 'abs'/'rel', absolute or relative joint-based trajectory representation
        - up_axis: y/z
    output:
        - r_rot_quat: joint-based root rotation
        - r_pos: joint-based root linear position
    '''
    if up_axis == 'y':
        up_axis_dim, face_axis_dim, quat_sin_dim = 1, 2, 2
    elif up_axis == 'z':
        up_axis_dim, face_axis_dim, quat_sin_dim = 2, 1, 3
    else:
        print('[ERROR] up_axis not setup correctly.')
        exit()

    if root_traj_repr == 'abs':
        r_rot_ang = data[..., 0]
        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., quat_sin_dim] = torch.sin(r_rot_ang)
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., [0, face_axis_dim]] = data[..., 1:3]
        r_pos[..., up_axis_dim] = data[..., 3]
    elif root_traj_repr == 'rel':
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get up-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., quat_sin_dim] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, face_axis_dim]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., up_axis_dim] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_repr_smpl(data_dict, recover_mode='joint_abs_traj', smplx_model=None, return_verts=False, return_full_joints=False):
    '''
    Recover joint/vertices positions from the motion representation
    input:
        - data_dict: motion representation
        - recover_mode:
            joint_abs_traj: recover from joint-based repr using absolute joint-based traj repr
            joint_rel_traj: recover from joint-based repr using relative joint-based traj repr
            smplx_params: recover from smplx-based repr
        - return_verts: return body vertices
        - return_full_joints: return all 127 smplx joints
    output:
        - smplx_joints: smplx joints, only include 22 main body joints if not return_full_joints
        - smplx_verts: smplx vertices
    '''
    if recover_mode not in ['joint_abs_traj', 'joint_rel_traj', 'smplx_params']:
        print('[ERROR] recover_mode incorrect! in func recover_from_repr_smpl()')
    if recover_mode[0:5] == 'joint':
        root_rot_angle = data_dict['root_rot_angle']
        root_rot_angle_vel = data_dict['root_rot_angle_vel']
        root_l_pos = data_dict['root_l_pos']
        root_l_vel = data_dict['root_l_vel']
        root_height = data_dict['root_height']
        if recover_mode == 'joint_abs_traj':
            root_traj_repr = torch.cat([root_rot_angle, root_l_pos, root_height], axis=-1)  # [..., 4]
            r_rot_quat, r_pos = recover_root_rot_pos(data=root_traj_repr, root_traj_repr='abs', up_axis='z')
        elif recover_mode == 'joint_rel_traj':
            root_traj_repr = torch.cat([root_rot_angle_vel, root_l_vel, root_height], axis=-1)  # [..., 4]
            r_rot_quat, r_pos = recover_root_rot_pos(data=root_traj_repr, root_traj_repr='rel', up_axis='z')

        positions = data_dict['local_positions'][..., 3:]  # [..., 21*3]
        positions = positions.view(positions.shape[:-1] + (-1, 3))  # [..., 21, 3]
        '''Add up-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
        '''Add root transl (projected on ground plane) to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 1] += r_pos[..., 1:2]
        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        return positions

    elif recover_mode == 'smplx_params':
        bs = len(data_dict['smplx_rot_6d'])
        global_orient_mat = rot6d_to_rotmat(data_dict['smplx_rot_6d'].reshape(-1, 6))  # [bs*T, 3, 3]
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat) # [bs*T, 3]
        body_pose_mat = rot6d_to_rotmat(data_dict['smplx_body_pose_6d'].reshape(-1, 6))  # [bs*T*21, 3, 3]
        body_pose_aa = rotation_matrix_to_angle_axis(body_pose_mat).reshape(-1, 21, 3)  # [bs*T, 21, 3]
        smplx_params_dict = {'transl': data_dict['smplx_trans'].reshape(-1, 3),
                             'global_orient': global_orient_aa,
                             'body_pose': body_pose_aa.reshape(-1, 63),
                             'betas': data_dict['smplx_betas'].reshape(-1, 10),
                             'jaw_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'leye_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'reye_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'left_hand_pose': torch.zeros(len(global_orient_aa), 45).to(global_orient_aa.device),
                             'right_hand_pose': torch.zeros(len(global_orient_aa), 45).to(global_orient_aa.device),
                             'expression': torch.zeros(len(global_orient_aa), 10).to(global_orient_aa.device)}
        smplx_output = smplx_model(**smplx_params_dict)
        if return_full_joints:
            smplx_joints = smplx_output.joints.reshape(bs, -1, 127, 3)
        else:
            smplx_joints = smplx_output.joints[:, 0:22].reshape(bs, -1, 22, 3)
        if return_verts:
            smplx_verts = smplx_output.vertices.reshape(bs, -1, 10475, 3)
            return smplx_joints, smplx_verts
        else:
            return smplx_joints
