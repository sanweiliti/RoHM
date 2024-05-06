import smplx
import os
import pickle
import pyrender
import trimesh
import numpy as np
import json
import torch
import argparse
from tqdm import tqdm
import cv2
import PIL.Image as pil_img

group = argparse.ArgumentParser(description='')
group.add_argument('--prox_root', type=str, default='/mnt/hdd/PROX', help='path to dataset')
group.add_argument('--body_model_path', type=str, default='../data/body_models/smplx_model', help='path to smplx model')
group.add_argument('--init_body_path', type=str, default='../data/init_motions/init_prox_rgbd', help='path to dataset')
group.add_argument('--save_mask_path', type=str, default='../mask_joint_prox', help='path to dataset')
group.add_argument('--scene_name', type=str, default='N0Sofa', help='path to dataset')
group.add_argument('--seq_name', type=str, default='N0Sofa_00034_01', help='path to dataset')

args = group.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_prox_pkl(pkl_path):
    body_params_dict = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        body_params_dict['transl'] = data['transl'][0]
        body_params_dict['global_orient'] = data['global_orient'][0]
        body_params_dict['betas'] = data['betas'][0]
        body_params_dict['body_pose'] = data['body_pose'][0]  # array, [63,]
        # body_params_dict['pose_embedding'] = data['pose_embedding'][0]
        body_params_dict['left_hand_pose'] = data['left_hand_pose'][0]
        body_params_dict['right_hand_pose'] = data['right_hand_pose'][0]
        body_params_dict['jaw_pose'] = data['jaw_pose'][0]
        body_params_dict['leye_pose'] = data['leye_pose'][0]
        body_params_dict['reye_pose'] = data['reye_pose'][0]
        body_params_dict['expression'] = data['expression'][0]
    return body_params_dict

def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                             np.asarray(cam['k']))[0].squeeze()


if __name__ == "__main__":
    prox_params_folder = '{}/{}'.format(args.init_body_path, args.seq_name)
    img_folder = '{}/recordings/{}/Color'.format(args.prox_root, args.seq_name)
    scene_mesh_path = '{}/scenes/{}.ply'.format(args.prox_root, args.scene_name)
    save_mask_folder = '{}/{}'.format(args.save_mask_path, args.seq_name)

    with open(os.path.join(args.prox_root, 'cam2world', args.scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))
    with open(os.path.join(args.prox_root, 'calibration', 'Color.json'), 'r') as f:
        color_cam = json.load(f)

    smplx_model = smplx.create(model_path=args.body_model_path, model_type="smplx", gender='neutral', flat_hand_mean=True, use_pca=False).to(device)
    print('[INFO] smplx model loaded.')

    ########### render settings
    camera_center = np.array([951.30, 536.77])
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera_render = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    static_scene = trimesh.load(scene_mesh_path)
    trans = np.linalg.inv(cam2world)
    static_scene.apply_transform(trans)
    static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

    ############################# redering scene #######################
    scene = pyrender.Scene()
    scene.add(camera_render, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(static_scene_mesh, 'mesh')
    r = pyrender.OffscreenRenderer(viewport_width=1920,
                                   viewport_height=1080)
    color_scene, depth_scene = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]
    # color_scene = color_scene.astype(np.float32) / 255.0
    # img_scene = (color_scene * 255).astype(np.uint8)

    ############# render, mask
    img_list = os.listdir(img_folder)
    img_list = sorted(img_list)
    seq_mask = []
    cnt = 0
    for img_fn in tqdm(img_list[0:100]):
        cnt += 1
        if img_fn.endswith('.png') or img_fn.endswith('.jpg') and not img_fn.startswith('.'):
            mask = np.ones([25])

            ######## get smplx body mesh
            prox_params_dir = os.path.join(prox_params_folder, 'results', img_fn[0:-4], '000.pkl')
            body_params_dict = {}
            with open(prox_params_dir, 'rb') as f:
                data_dict = pickle.load(f)
            smplx_params_dict = {'transl': torch.from_numpy(data_dict['transl']).to(device),
                                 'global_orient': torch.from_numpy(data_dict['global_orient']).to(device),
                                 'body_pose': torch.from_numpy(data_dict['body_pose']).to(device),
                                 'betas': torch.from_numpy(data_dict['betas']).to(device),
                                 'jaw_pose': torch.zeros(1, 3).to(device),
                                 'leye_pose': torch.zeros(1, 3).to(device),
                                 'reye_pose': torch.zeros(1, 3).to(device),
                                 'left_hand_pose': torch.zeros(1, 45).to(device),
                                 'right_hand_pose': torch.zeros(1, 45).to(device),
                                 'expression': torch.zeros(1, 10).to(device)}
            smplx_output = smplx_model(return_verts=True, **smplx_params_dict)  # generated human body mesh
            body_verts = smplx_output.vertices.detach().cpu().numpy()[0]
            body_mesh = trimesh.Trimesh(body_verts, smplx_model.faces, process=False)
            body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=material)

            ############################# rendering body #######################
            scene = pyrender.Scene()
            scene.add(camera_render, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            scene.add(body_mesh, 'mesh')
            r = pyrender.OffscreenRenderer(viewport_width=1920,
                                           viewport_height=1080)
            color_body, depth_body = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]

            ######### body joints --> set mask
            body_joints_3d = smplx_output.joints.detach().cpu().numpy()  # [1, n, 3]
            projected_joints = projectPoints(body_joints_3d.reshape(-1, 3), color_cam)
            projected_joints = projected_joints.reshape(body_joints_3d.shape[0], -1, 2)  # # [1, n, 2]
            projected_joints = projected_joints[0][0:25]  # [25, 2]
            projected_joints = projected_joints.astype(int)
            # for 25 smplx main body joints
            for j_id in range(25):
                x_coord, y_coord = projected_joints[j_id][0], projected_joints[j_id][1]
                if 0 <= x_coord < 1920 and 0 <= y_coord < 1080:
                    if depth_body[y_coord][x_coord] - depth_scene[y_coord][x_coord] > 0.1 \
                            and depth_scene[y_coord][x_coord] != 0:  # todo: set threshold
                        mask[j_id] = 0  # occlusion happens, mask corresponding joint
            seq_mask.append(mask)

            # ############################# visualization #########################
            # #############  render body+scene
            # scene = pyrender.Scene()
            # scene.add(camera_render, pose=camera_pose)
            # scene.add(light, pose=camera_pose)
            # scene.add(static_scene_mesh, 'mesh')
            # scene.add(body_mesh, 'mesh')
            # r = pyrender.OffscreenRenderer(viewport_width=1920,
            #                                viewport_height=1080)
            # color, _ = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]
            # color = color.astype(np.float32) / 255.0
            # save_img = (color * 255).astype(np.uint8)
            #
            # ########## draw 2d joints and occlusion masks (green: occluded, red: visible)
            # for k in range(len(projected_joints)):
            #     for p in range(max(0, projected_joints[k][0] - 3), min(1920 - 1, projected_joints[k][0] + 3)):
            #         for q in range(max(0, projected_joints[k][1] - 3), min(1080 - 1, projected_joints[k][1] + 3)):
            #             if mask[k] == 1:
            #                 save_img[q][p][0] = 255
            #                 save_img[q][p][1] = 0
            #                 save_img[q][p][2] = 0
            #             else:
            #                 save_img[q][p][0] = 0
            #                 save_img[q][p][1] = 255
            #                 save_img[q][p][2] = 0
            #
            # ######### save img (for visualization)
            # save_img = pil_img.fromarray(save_img.astype(np.uint8))
            # # save_img.show()
            # # save_img.save('{}/{}'.format(save_img_folder, img_fn))

    if not os.path.exists(save_mask_folder):
        os.makedirs(save_mask_folder)
    seq_mask = np.asarray(seq_mask)
    np.save('{}/mask_joint.npy'.format(save_mask_folder), seq_mask)







