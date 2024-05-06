import trimesh
import pyrender
import PIL.Image as pil_img
from utils.other_utils import *

contact_idx_dic = {7: 0, 10: 1, 8: 2, 11: 3}
material_body_rec_vis = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(66 / 255, 149 / 255, 245 / 255, 1.0)  # light blue
    )
material_body_rec_occ = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(212 / 255, 189 / 255, 102 / 255, 1.0)  # light yellow
)
material_body_noisy = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(198/255, 226/255, 255/255, 1.0)  # white
    )
material_body_gt = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(255 / 255, 102 / 255, 102 / 255, 1.0)  # light red
    )
material_joint_vis = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(6 / 255, 75 / 255, 255 / 255, 1.0)  # blue
    )
material_joint_occ = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(222 / 255, 177 / 255, 4 / 255, 1.0)  # yellow 222, 177, 4
    )
material_skel_vis = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(90 / 255, 135 / 255, 247 / 255, 1.0)  # light blue  134, 166, 247
    )
material_skel_occ = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(219 / 255, 199 / 255, 123 / 255, 1.0)  # light yellow 219, 199, 123
    )
material_contact_1 = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 139 / 255, 0.0, 1.0)  # green
    )
material_contact_0 = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(205 / 255, 0.0, 0.0, 1.0)  # red
    )

def create_render_cam(cam_x, cam_y, fx, fy):
    # H, W = 1080, 1920
    camera_center = np.array([cam_x, cam_y])
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    return camera, camera_pose, light

##################### get floor plane
def create_floor(trans,):
    color0 = [0.8, 0.9, 0.9]
    color1 = [0.6, 0.7, 0.7]
    alpha = 1.0
    tile_width = 0.5
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    # make checkerboard
    length = 25.0
    radius = length / 2.0
    num_rows = num_cols = int(length / tile_width)
    vertices = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            start_loc = [-radius + j * tile_width, radius - i * tile_width]
            cur_verts = np.array([[start_loc[0], start_loc[1], 0.0],
                                  [start_loc[0], start_loc[1] - tile_width, 0.0],
                                  [start_loc[0] + tile_width, start_loc[1] - tile_width, 0.0],
                                  [start_loc[0] + tile_width, start_loc[1], 0.0]])
            cur_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=int)
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_face_colors = np.array([cur_color, cur_color])
            vertices.append(cur_verts)
            faces.append(cur_faces)
            face_colors.append(cur_face_colors)
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    face_colors = np.concatenate(face_colors, axis=0)
    ground_tri = trimesh.creation.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors, process=False)
    ground_tri.apply_transform(np.linalg.inv(trans))
    ground_mesh = pyrender.Mesh.from_trimesh(ground_tri, smooth=False)
    return ground_mesh

def create_pyrender_scene(camera, camera_pose, light):
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    return scene

def create_pyrender_mesh(verts, faces, trans, material=None, vertex_colors=None, ):
    body = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors, process=False)
    body.apply_transform(np.linalg.inv(trans))
    body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
    return body_mesh

def create_pyrender_skel(joints, add_trans=None,
                         mask_scheme='', mask_joint_id=None, add_occ_joints=True,
                         t=0, start=0, end=0,
                         add_contact=False, contact_lbl=None):
    skeleton_mesh_list = []
    for j in range(22):
        sphere = trimesh.creation.icosphere(radius=0.025)
        transformation = np.identity(4)
        transformation[:3, 3] = joints[j]
        sphere.apply_transform(transformation)
        sphere.apply_transform(add_trans) if add_trans is not None else None
        if add_contact and j in [7, 10, 8, 11]:
            material = material_contact_1 if contact_lbl[contact_idx_dic[j]] == 1 else material_contact_0
        else:
            if mask_scheme in ['lower', 'video']:
                material = material_joint_vis if j not in mask_joint_id else material_joint_occ
            elif mask_scheme == 'full':
                material = material_joint_vis if (t < start or t >= end) else material_joint_occ
        if mask_scheme != 'full':
            if add_occ_joints or (not add_occ_joints and j not in mask_joint_id):
                sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
                skeleton_mesh_list.append(sphere_mesh)
        else:
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            skeleton_mesh_list.append(sphere_mesh)

    for index_pair in LIMBS_BODY_SMPL:
        p1 = joints[index_pair[0]]
        p2 = joints[index_pair[1]]
        segment = np.array([p1, p2])
        cyl = trimesh.creation.cylinder(0.01, height=None, segment=segment)
        cyl.apply_transform(add_trans)
        if mask_scheme in ['lower', 'video']:
            material = material_skel_vis if (index_pair[0] not in mask_joint_id and index_pair[1] not in mask_joint_id) else material_joint_occ
        elif mask_scheme == 'full':
            material = material_skel_vis if (t < start or t >= end) else material_joint_occ
        if not (mask_scheme in ['lower', 'video'] and not add_occ_joints and (index_pair[0] in mask_joint_id or index_pair[1] in mask_joint_id)):
            cyl_mesh_rec = pyrender.Mesh.from_trimesh(cyl, material=material)
            skeleton_mesh_list.append(cyl_mesh_rec)
    return skeleton_mesh_list


def render_img(renderer, scene, alpha=1.0):
    # alpha: transparency in [0, 1]
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    color[:, :, -1] = color[:, :, -1] * alpha
    color = pil_img.fromarray((color * 255).astype(np.uint8))
    return color

def render_img_overlay(renderer, scene, input_img, alpha=1.0):
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
    output_img = pil_img.fromarray((output_img).astype(np.uint8))
    return output_img