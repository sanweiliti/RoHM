import numpy as np
import cv2
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import datetime
import os, json, sys
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R



####### configs for motion representation: joint-based + smplx_based
REPR_LIST = ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel', 'root_height', # joint-based traj
             'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel',  # smplx-based traj
             'local_positions', 'local_vel',  # joint-based local pose
             'smplx_body_pose_6d',  # smplx-based local pose
             'smplx_betas',  # smplx body shape
             'foot_contact', ]  # [foot contact labels]
# dimension for each categody of the motion representation
REPR_DIM_DICT = {'root_rot_angle': 1,
                 'root_rot_angle_vel': 1,
                 'root_l_pos': 2,
                 'root_l_vel': 2,
                 'root_height': 1,
                 'smplx_rot_6d': 6,
                 'smplx_rot_vel': 3,
                 'smplx_trans': 3,
                 'smplx_trans_vel': 3,
                 'local_positions': 22 * 3,
                 'local_vel': 22 * 3,
                 'smplx_body_pose_6d': 21 * 6,
                 'smplx_betas': 10,
                 'foot_contact': 4, }

# estimated floor height from scene mesh
# z axis up
prox_floor_height = {'N0Sofa': -0.9843093165454873,
                     'MPH1Library': -0.34579620031341207,
                     'N3Library': -0.6736229583361132,
                     'N3Office': -0.7772727989022952,
                     'BasementSittingBooth': -0.767080139846674,
                     'MPH8': -0.41432886722717904,
                     'MPH11': -0.7169139211234009,
                     'MPH16': -0.8408992040141058,
                     'MPH112': -0.6419028605753081,
                     'N0SittingBooth': -0.6677103008966809,
                     'N3OpenArea': -1.0754909672969915,
                     'Werkraum': -0.6777057869851316}

# y axis up
egobody_floor_height = {'seminar_g110': -1.660,
                        'seminar_d78': -0.810,
                        'seminar_j716': -0.8960,
                        'seminar_g110_0315': -0.73,
                        'seminar_d78_0318': -1.03,
                        'seminar_g110_0415': -0.77}

LIMBS_BODY_SMPL = [(15, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         # (20, 22),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         # (21, 23),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11),]

def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                             np.asarray(cam['k']))[0].squeeze()

def perspective_projection(points,
                           # translation,
                           focal_length,
                           camera_center=None,
                           rotation=None):
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    # points = points + translation.unsqueeze(1)
    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]



def update_globalRT_for_smplx(body_param_dict, trans_to_target_origin, smplx_model=None, device=None, delta_T=None):
    '''
    input:
        body_param_dict:
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
        delta_T: pelvis location?
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    bs = len(body_param_dict['transl'])

    if delta_T is None:
        body_param_dict_torch = {}
        for key in body_param_dict.keys():
            body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(device)
        body_param_dict_torch['transl'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)

        body_param_dict_torch['jaw_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['leye_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['reye_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['left_hand_pose'] = torch.zeros(bs, 45).to(device)
        body_param_dict_torch['right_hand_pose'] = torch.zeros(bs, 45).to(device)
        body_param_dict_torch['expression'] = torch.zeros(bs, 10).to(device)

        smpl_out = smplx_model(**body_param_dict_torch)
        delta_T = smpl_out.joints[:,0,:] # (bs, 3,)
        delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict['global_orient']
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [bs, 3,3] rotation mat
    body_T = body_param_dict['transl']
    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1,:-1] = body_R_mat
    body_mat[:, :-1, -1] = body_T + delta_T
    body_mat[:, -1, -1] = 1

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    trans_to_target_origin = np.expand_dims(trans_to_target_origin, axis=0)  # [1, 4]
    trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)  # [bs, 4]

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    body_R_new = R.from_matrix(body_mat_new[:, :-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:, :-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(-1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(-1,3)
    return body_params_dict_new


def estimate_angular_velocity(rot_seq, dRdt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    # dRdt = self.estimate_linear_velocity(rot_seq, h)

    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], dim=-1)  # [B, T, ..., 3]
    return w


def estimate_angular_velocity_np(rot_seq, dRdt):
    # rot_seq: [T, 3, 3]
    # dRdt: [T, 3, 3]
    R = rot_seq
    RT = np.transpose(R, (0, -1, -2))
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)  # [B, T, ..., 3]
    return w


