import sys, os
import torch
import numpy as np
from tqdm import tqdm
# from utils import *
import glob
import argparse
import smplx


############################
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ex_fps = 30


def amass_to_pose(dataset_name, src_path, save_path_joints, save_path_smplx_params, smplx_model_neutral):
    bdata = np.load(src_path, allow_pickle=True)
    fps = bdata['mocap_frame_rate']
    # print('fps:', fps)
    frame_number = bdata['trans'].shape[0]
    process = True

    if bdata['gender'] != 'neutral':
        print('gender not neutral!')
        process = False
    if bdata['surface_model_type'] != 'smplx':
        print('not smplx params!')
        process = False

    # for SSM: fps=59.99xx/120.00xx
    if dataset_name == 'SSM':
        if fps - 60 < 1:
            down_sample = 2
        else:
            down_sample = 4
    else:
        down_sample = int(fps / ex_fps)
        if down_sample != fps / ex_fps:
            process = False
            print('frame rate {} not suitable for dowmsampling to 30fps.'.format(fps))

    pose_seq = []
    smplx_params_seq = []
    if process:
        # print('process:', process, 'fps:', fps, 'down_sample', down_sample, 'total_frame_number', frame_number)
        with torch.no_grad():
            for fId in range(0, frame_number, down_sample):
                global_orient = torch.Tensor(bdata['root_orient'][fId:fId + 1, :]).to(comp_device)  # [1, 3]
                trans = torch.Tensor(bdata['trans'][fId:fId + 1]).to(comp_device)  # [1, 3]
                betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)  # [1, 10]
                body_pose = torch.Tensor(bdata['pose_body'][fId:fId + 1, :]).to(comp_device)  # [1, 63]
                hand_pose = torch.Tensor(bdata['pose_hand'][fId:fId + 1, :]).to(comp_device)  # [1, 90]
                jaw_pose = torch.Tensor(bdata['pose_jaw'][fId:fId + 1, :]).to(comp_device)  # [1, 3]
                leye_pose = torch.Tensor(bdata['pose_eye'][fId:fId + 1, 0:3]).to(comp_device)  # [1, 3]
                reye_pose = torch.Tensor(bdata['pose_eye'][fId:fId + 1, 0:3]).to(comp_device)  # [1, 3]

                body_params = {'global_orient': global_orient,
                               'transl': trans,
                               'betas': betas,
                               'body_pose': body_pose,
                               'hand_pose': hand_pose,
                               'jaw_pose': jaw_pose,
                               'leye_pose': leye_pose,
                               'reye_pose': reye_pose,}
                smplx_output = smplx_model_neutral(**body_params)
                smplx_joints = smplx_output.joints[:, 0:25]  # [1, 25, 3]

                smplx_params_seq.append(torch.cat([global_orient, trans, betas, body_pose, hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1))  # [1, 169]
                pose_seq.append(smplx_joints)

        pose_seq = torch.cat(pose_seq, dim=0)
        pose_seq_np = pose_seq.detach().cpu().numpy()  # [seq_len, 52, 3], position of 52 body joints?

        smplx_params_seq = torch.cat(smplx_params_seq, dim=0)  # [seq_len, 169]
        smplx_params_seq_np = smplx_params_seq.detach().cpu().numpy()  # [seq_len, 169],

        np.save(save_path_joints, pose_seq_np)
        np.save(save_path_smplx_params, smplx_params_seq_np)
    return fps

# dataset_name:
# ACCAD  --> fps=120
# BMLmovi  --> fps=120
# BMLrub  --> fps=120
# CMU  --> fps=60/120
# CNRS  --> fps=100
# DFaust  --> fps=120
# EKUT  --> fps=100
# Eyes_Japan_Dataset  --> fps=120/250
# GRAB --> fps=120
# HDM05 --> fps=120
# HumanEva --> fps=120
# KIT  --> fps=100
# MoSh --> fps=100/120
# PosePrior --> fps=120
# SFU --> fps=120
# SOMA --> fps=120
# SSM --> fps=59.99xx/120.00xx
# TCDHands --> fps=120
# TotalCapture --> fps=60
# Transitions --> fps=120
# WEIZMANN  --> fps=100

def main(args):
    smplx_model_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx", gender='neutral').to(comp_device)

    print('datasets in process:', args.dataset_name)
    subj_list = [x for x in os.listdir(os.path.join(args.amass_root, args.dataset_name)) if os.path.isdir(os.path.join(args.amass_root, args.dataset_name, x))]
    subj_list = sorted(subj_list)

    ############################
    save_root_joints = os.path.join(args.save_root, 'pose_data_fps_{}'.format(ex_fps))
    save_root_smpl_params = os.path.join(args.save_root, 'smpl_data_fps_{}'.format(ex_fps))
    save_folders = [os.path.join(save_root_joints, args.dataset_name, subj) for subj in subj_list]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    save_folders = [os.path.join(save_root_smpl_params, args.dataset_name, subj) for subj in subj_list]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)

    npz_path_list = glob.glob(os.path.join(args.amass_root, args.dataset_name, '*/*.npz'))
    npz_path_list = sorted(npz_path_list)
    for npz_path in tqdm(npz_path_list):
        recording_name = npz_path.split('/')[-1][0:-4]
        subj = npz_path.split('/')[-2]
        # print(recording_name)
        if recording_name == 'neutral_stagei':
            continue
        ######## remove ice skating clips from HDM05: dg/HDM_dg_07-01* is inline skating
        if args.dataset_name == 'HDM05' and recording_name[0:12] == 'HDM_dg_07-01':
            continue
        ######## remove treadmill clips from BMLrub
        if args.dataset_name == 'BMLrub' and (recording_name.split('_')[1] == 'treadmill' or recording_name.split('_')[1] == 'normal'):
            continue
        ######## process sequence
        save_path_joints = os.path.join(save_root_joints, args.dataset_name, subj, recording_name + '.npy')
        save_path_smpl_params = os.path.join(save_root_smpl_params, args.dataset_name, subj, recording_name + '.npy')
        fps = amass_to_pose(args.dataset_name, npz_path, save_path_joints, save_path_smpl_params, smplx_model_neutral)

    print('finished.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_model_path', type=str, default='data/body_models/smplx_model', help='path to smplx model')
    parser.add_argument('--amass_root', type=str, default='/mnt/hdd/AMASS/AMASS_smplx_neutral', help='Root dir of raw AMASS data (smplx neutral body)')
    parser.add_argument('--dataset_name', type=str, default='ACCAD', help='AMASS subset name')
    # AMASS_preprocessed
    parser.add_argument('--save_root', type=str, default='/mnt/hdd/AMASS/AMASS_smplx_preprocessed', help='Root directory to save preprocessed data to.')

    config = parser.parse_known_args()
    config = config[0]

    main(config)






