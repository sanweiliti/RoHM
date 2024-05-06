import open3d as o3d
from utils.other_utils import *

COLOR_VIS_O3D = [90 / 255, 135 / 255, 247 / 255]
COLOR_OCC_O3D = [219 / 255, 199 / 255, 123 / 255]
COLOR_GT_O3D = [255 / 255, 102 / 255, 102 / 255]
COLOR_CONTACT_1 = [0 / 255, 128 / 255, 0 / 255]
COLOR_CONTACT_0 = [128 / 255, 0 / 255, 0 / 255]


def vis_skeleton(joints, limbs,
                 add_trans=None,
                 mask_scheme=None, cur_mask_joint_id=None,
                 start=0, end=0, t=0,
                 color_occ=COLOR_OCC_O3D, color_vis=COLOR_VIS_O3D):
    """
    Input:
        joints: numpy [n_joints, 3], joint positions 
        limbs: limb topology
        add_trans: numpy [3], additional translation for skeleton
        mask_scheme: occlusion mask scheme, 'lower'/'full'/'video'
        start/end: start/end frame for full-body occlusion mask if mask_scheme=='full'
        t: current timestep (for full-body occlusion visualization)
    Output:
        skeleton_list: open3d body skeleton (a list of open3d arrows)
    """
    skeleton_list = []
    for limb in limbs:
        bone_length = np.linalg.norm(joints[[limb[0]]] - joints[[limb[1]]], axis=-1)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.03, cone_radius=0.001,
                                                       cylinder_height=bone_length, cone_height=0.001)
        mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), joints[limb[1]] - joints[limb[0]])
        transformation = np.identity(4)
        transformation[0:3, 0:3] = mat
        transformation[:3, 3] = joints[limb[0]]
        if add_trans is not None:
            transformation[0:3, 3] += add_trans
        arrow.transform(transformation)
        if mask_scheme is None:
            arrow.paint_uniform_color(COLOR_GT_O3D)
        elif mask_scheme in ['lower', 'video']:
            if limb[0] in cur_mask_joint_id or limb[1] in cur_mask_joint_id:
                arrow.paint_uniform_color(color_occ)
            else:
                arrow.paint_uniform_color(color_vis)
        elif mask_scheme == 'full':
            if t >= start and t < end:
                arrow.paint_uniform_color(color_occ)
            else:
                arrow.paint_uniform_color(color_vis)
        else:
            print('[ERROR] mask_scheme {} not defined.'.format(mask_scheme))
            exit()
        arrow.compute_vertex_normals()
        skeleton_list.append(arrow)
    return skeleton_list


def vis_foot_contact(joints, contact_lbl, add_trans=None):
    """
    Input:
        joints: numpy [n_joints, 3], joint positions
        contact_lbl: numpy, [4]
        add_trans: numpy [3], additional translation for skeleton
    Output:
        foot_sphere_list: sphere of foot joints with color denoting contact labels (a list of open3d spheres)
    """
    foot_sphere_list = []
    for idx, foot_joint_idx in enumerate([7, 10, 8, 11]):
        transformation = np.identity(4)
        transformation[:3, 3] = joints[foot_joint_idx]
        if add_trans is not None:
            transformation[0:3, 3] += add_trans
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        if contact_lbl[idx] == 1:
            sphere.paint_uniform_color(COLOR_CONTACT_1)  # green, in contact
        else:
            sphere.paint_uniform_color(COLOR_CONTACT_0)  # red, not in contact
        sphere.transform(transformation)
        foot_sphere_list.append(sphere)
    return foot_sphere_list