# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import argparse
import matplotlib
import pprint
import glob
import copy
import posecnn_cuda

import os
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
import torch
import cv2
import matplotlib.pyplot as plt
import scipy.io
import yaml
from transforms3d.quaternions import quat2mat, mat2quat, quat2axangle

from pose_rbpf.pose_rbpf import *
from pose_rbpf.sdf_multiple_optimizer import sdf_multiple_optimizer
from datasets.ycb_video_dataset import *
from datasets.dex_ycb_dataset import *
from datasets.tless_dataset import *
from config.config import cfg, cfg_from_file
from utils.cython_bbox import bbox_overlaps

posecnn_classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', \
                   '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', \
                   '011_banana', '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', \
                   '036_wood_block', '037_scissors', '040_large_marker', '052_extra_large_clamp', '061_foam_brick')

# demo_dir = "/media/gdk/Data/Datasets/kitchen_data/20220513_extrinsics/1652457909/827312071624"

# obj_map = np.array([1, 12, 3, 4, 14, 11, 13, 20, 16])
obj_map = np.array([13, 14])
model_path = '/media/gdk/Hard_Disk/Datasets/dex_data/models'
obj_file = {1: f'{model_path}/002_master_chef_can/textured_simple.obj', 2: f'{model_path}/003_cracker_box/textured_simple.obj', 3: f'{model_path}/004_sugar_box/textured_simple.obj', 4: f'{model_path}/005_tomato_soup_can/textured_simple.obj', 5: f'{model_path}/006_mustard_bottle/textured_simple.obj', 6: f'{model_path}/007_tuna_fish_can/textured_simple.obj', 7: f'{model_path}/008_pudding_box/textured_simple.obj', 8: f'{model_path}/009_gelatin_box/textured_simple.obj', 9: f'{model_path}/010_potted_meat_can/textured_simple.obj', 10: f'{model_path}/011_banana/textured_simple.obj', 11: f'{model_path}/019_pitcher_base/textured_simple.obj', 12: f'{model_path}/021_bleach_cleanser/textured_simple.obj', 13: f'{model_path}/024_bowl/textured_simple.obj', 14: f'{model_path}/025_mug/textured_simple.obj', 15: f'{model_path}/035_power_drill/textured_simple.obj', 16: f'{model_path}/036_wood_block/textured_simple.obj', 17: f'{model_path}/037_scissors/textured_simple.obj', 18: f'{model_path}/040_large_marker/textured_simple.obj', 19: f'{model_path}/051_large_clamp/textured_simple.obj', 20: f'{model_path}/052_extra_large_clamp/textured_simple.obj', 21: f'{model_path}/061_foam_brick/textured_simple.obj'}


def create_scene(intrinsics, obj_map):
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = intrinsics[0, 0]
  fy = intrinsics[1, 1]
  px = intrinsics[0, 2]
  py = intrinsics[1, 2]
  cam = pyrender.IntrinsicsCamera(fx, fy, px, py)
  # scene.add(cam, pose=np.eye(4))

  # Load YCB meshes.
  meshes = {}
  for i in obj_map:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    meshes[i] = mesh

  return scene, meshes, cam


def render_pose(im_file, scene, meshes, cam, poses, ycb_ids):
    scene.add(cam, pose=np.eye(4))

    for pose, id in zip(poses, ycb_ids):
      if np.all(pose == 0.0):
        continue
      pose = np.vstack((pose, np.array([[0, 0, 0, 1]], dtype=np.float32)))
      pose[1] *= -1
      pose[2] *= -1

      node = scene.add(meshes[id], pose=pose)

      # scene.set_pose(nodes[id], pose)

    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    im_render, _ = r.render(scene)

    im_real = cv2.imread(im_file)
    im_real = im_real[:, :, ::-1]

    im = 0.5 * im_real.astype(np.float32) + 0.5 * im_render.astype(np.float32)
    im = im.astype(np.uint8)

    scene.clear()

    return im


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test PoseRBPF on YCB Video or T-LESS Datasets (RGBD)')
    parser.add_argument('--test_config', dest='test_cfg_file',
                        help='configuration for testing',
                        required=True, type=str)
    parser.add_argument('--pf_config_dir', dest='pf_cfg_dir',
                        help='directory for poserbpf configuration files',
                        default='./config/test/YCB/', type=str)
    parser.add_argument('--train_config_dir', dest='train_cfg_dir',
                        help='directory for AAE training configuration files',
                        default='./checkpoints/ycb_configs_roi_rgbd/', type=str)
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        help='directory for AAE ckpts',
                        default='./checkpoints/ycb_ckpts_roi_rgbd/', type=str)
    parser.add_argument('--codebook_dir', dest='codebook_dir',
                        help='directory for codebooks',
                        default='./checkpoints/ycb_codebooks_roi_rgbd/', type=str)
    parser.add_argument('--modality', dest='modality',
                        help='modality',
                        default='rgbd', type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to test on',
                        default='ycb_video', type=str)
    parser.add_argument('--dataset_dir', dest='dataset_dir',
                        help='relative dir of the dataset',
                        default='../YCB_Video_Dataset/data/',
                        type=str)
    parser.add_argument('--cad_dir', dest='cad_dir',
                        help='directory of objects CAD models',
                        default='./cad_models',
                        type=str)
    parser.add_argument('--n_seq', dest='n_seq',
                        help='index of sequence',
                        default=1,
                        type=int)
    parser.add_argument('--demo', dest='demo',
                        help='run as demo mode',
                        default=False,
                        type=bool)
    parser.add_argument('--depth_refinement', dest='refine',
                        help='sdf refinement',
                        action='store_true')
    parser.add_argument('--demo_dir', dest='demo_dir',
                        help='directory for demo',
                        default='', type=str)
    args = parser.parse_args()
    return args


def get_images(color_file, depth_file):

    # rgba
    rgba = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
    if rgba.shape[2] == 4:
        im = np.copy(rgba[:,:,:3])
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 0
    else:
        im = rgba
    im_color = im.astype('float') / 255.0

    # depth image
    im_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    im_depth = im_depth.astype('float') / 1000.0

    return im_color[:, :, (2, 1, 0)], im_depth


if __name__ == '__main__':

    args = parse_args()

    print(args)
    demo_dir = args.demo_dir
    # load the configurations
    test_cfg_file = args.test_cfg_file
    cfg_from_file(test_cfg_file)

    # load the test objects
    print('Testing with objects: ')
    print(cfg.TEST.OBJECTS)
    obj_list = cfg.TEST.OBJECTS


    object_category = 'ycb'
    with open('/home/gdk/Repositories/Pose_RBPF/PoseRBPF/datasets/ycb_video_classes.txt', 'r') as class_name_file:
        obj_list_all = class_name_file.read().split('\n')

    # pf config files
    pf_config_files = sorted(glob.glob(args.pf_cfg_dir + '*yml'))
    cfg_list = []
    for obj in obj_list:
        obj_idx = obj_list_all.index(obj)
        print(obj, obj_idx)
        train_config_file = args.train_cfg_dir + '{}.yml'.format(obj)
        pf_config_file = pf_config_files[obj_idx]
        cfg_from_file(train_config_file)
        cfg_from_file(pf_config_file)
        cfg_list.append(copy.deepcopy(cfg))
    print('%d cfg files' % (len(cfg_list)))

    # checkpoints and codebooks
    checkpoint_list = []
    codebook_list = []
    for obj in obj_list:
        if args.modality == 'rgbd':
            checkpoint_list.append(args.ckpt_dir+'{}_py3.pth'.format(obj))
        else:
            checkpoint_list.append(args.ckpt_dir+'{}.pth'.format(obj))
        if not os.path.exists(args.codebook_dir):
            os.makedirs(args.codebook_dir)
        codebook_list.append(args.codebook_dir+'{}.pth'.format(obj))
    print('checkpoint files:', checkpoint_list)
    print('codebook files:', codebook_list)

    # dataset

    dataloader = []
    j = 0

    meta_path = os.path.join(demo_dir, 'camera_meta.yml')
    with open(meta_path, "r") as stream:
        intrinsic_matrix = np.array(yaml.safe_load(stream)['INTRINSICS']).reshape(3, 3)

    color_names = sorted(glob.glob(f'{demo_dir}/rgb/*.png'))
    depth_names = sorted(glob.glob(f'{demo_dir}/depth/*.png'))

    for i, color_name in enumerate(color_names):
        # print(color_name)
        sample = {}
        sample['image_color_path'] = color_name
        sample['image_depth_path'] = depth_names[i]

        # Load PoseCNN result
        result = scipy.io.loadmat(os.path.join(demo_dir, f'object_pose/posecnn_results/{i:06d}.png.mat'))
        sample['labels_result'] = result['labels'].copy()
        sample['rois_result'] = result['rois'].copy()
        # change to rbpf index
        remove = []
        for i in range(len(sample['rois_result'])):
            # print(len(sample['rois_result']))
            # print(sample['rois_result'])
            # print(i)
            # print(sample['rois_result'][i])
            if sample['rois_result'][i][1] == 0 or not sample['rois_result'][i][1] in obj_map:
                remove.append(i)
            else:
                # print(sample['rois_result'][i][1])
                sample['rois_result'][i][1] = np.where(obj_map == sample['rois_result'][i][1])[0][0]

        sample['rois_result'] = np.delete(sample['rois_result'], remove, 0)

        sample['intrinsic_matrix'] = intrinsic_matrix
        sample['image_id'] = j
        j += 1
        sample['video_id'] = 0

        dataloader.append(sample)

    print('dataloader:', len(dataloader))


    # setup the poserbpf
    pose_rbpf = PoseRBPF(obj_list, cfg_list, checkpoint_list, codebook_list,
        object_category, modality=args.modality, cad_model_dir=args.cad_dir, gpu_id=args.gpu_id)

    if args.refine:
        print('loading SDFs')
        sdf_files = []
        for cls in obj_list:
            sdf_file = '{}/ycb_models/{}/textured_simple_low_res.pth'.format(args.cad_dir, cls)
            sdf_files.append(sdf_file)
        reg_trans = 1000.0
        reg_rot = 10.0
        sdf_optimizer = sdf_multiple_optimizer(obj_list, sdf_files, reg_trans, reg_rot)

    # output directory
    output_dir = os.path.join(demo_dir, 'object_pose/PoseRBPF')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #'''
    # loop the dataset
    visualize = False
    video_id = ''
    epoch_size = len(dataloader)
    for k, sample in enumerate(dataloader):
        # prepare data
        print(sample['image_color_path'])
        image_input, image_depth = get_images(sample['image_color_path'], sample['image_depth_path'])

        image_input = torch.from_numpy(image_input)
        im_depth = torch.from_numpy(image_depth).float()
        image_depth = torch.from_numpy(image_depth).unsqueeze(2)
        image_label = torch.from_numpy(sample['labels_result'])
        if image_label.shape[0] == 0:
            image_label = None
        width = image_input.shape[1]
        height = image_input.shape[0]
        intrinsics = sample['intrinsic_matrix']
        image_id = sample['image_id']

        # start a new video
        if video_id != sample['video_id']:
            pose_rbpf.reset_poserbpf()
            pose_rbpf.set_intrinsics(intrinsics, width, height)
            video_id = sample['video_id']
            print('start video %s' % (video_id))
            print(intrinsics)

        print('video %s, frame %s' % (video_id, image_id))


        print('instance', pose_rbpf.instance_list)
        print('ok_list', pose_rbpf.rbpf_ok_list)

        # detection from posecnn
        rois = sample['rois_result']

        # collect rois from rbpfs
        rois_rbpf = np.zeros((0, 6), dtype=np.float32)
        index_rbpf = []
        for i in range(len(pose_rbpf.instance_list)):
            if pose_rbpf.rbpf_ok_list[i]:
                roi = pose_rbpf.rbpf_list[i].roi
                rois_rbpf = np.concatenate((rois_rbpf, roi), axis=0)
                index_rbpf.append(i)
                pose_rbpf.rbpf_list[i].roi_assign = None

        # data association based on bounding box overlap
        num_rois = rois.shape[0]
        num_rbpfs = rois_rbpf.shape[0]
        print(rois, rois_rbpf)
        assigned_rois = np.zeros((num_rois, ), dtype=np.int32)
        if num_rbpfs > 0 and num_rois > 0:
            # overlaps: (rois x gt_boxes) (batch_id, x1, y1, x2, y2)
            overlaps = bbox_overlaps(np.ascontiguousarray(rois_rbpf[:, (1, 2, 3, 4, 5)], dtype=np.float),
                np.ascontiguousarray(rois[:, (1, 2, 3, 4, 5)], dtype=np.float))

            print('overlaps', overlaps)

            # assign rois to rbpfs
            assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)

            print(assignment)
            print(max_overlaps)
            unassigned = []
            for i in range(num_rbpfs):
                if max_overlaps[i] > 0.2:
                    pose_rbpf.rbpf_list[index_rbpf[i]].roi_assign = rois[assignment[i]]
                    assigned_rois[assignment[i]] = 1
                else:
                    unassigned.append(i)

            # check if there are un-assigned rois
            index = np.where(assigned_rois == 0)[0]

            # if there is un-assigned rbpfs
            if len(unassigned) > 0 and len(index) > 0:
                for i in range(len(unassigned)):
                    for j in range(len(index)):
                        if assigned_rois[index[j]] == 0 and pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi[0, 1] == rois[index[j], 1]:
                            pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi_assign = rois[index[j]]
                            assigned_rois[index[j]] = 1
        elif num_rbpfs == 0 and num_rois == 0:
            continue

        # filter tracked objects
        print(pose_rbpf.instance_list)
        print(pose_rbpf.rbpf_ok_list)
        for i in range(len(pose_rbpf.instance_list)):
            if pose_rbpf.rbpf_ok_list[i]:
                roi = pose_rbpf.rbpf_list[i].roi_assign
                print("here 1")
                print(roi)
                Tco, max_sim = pose_rbpf.pose_estimation_single(i, roi, image_input, image_depth, visualize=visualize)
                if max_sim < 0.75:
                    Tco, max_sim = pose_rbpf.pose_estimation_single(i, roi, image_input, image_depth, visualize=visualize)

                if roi is None:
                    pose_rbpf.rbpf_ok_list[i] = False

        # initialize new object
        print(assigned_rois)
        for i in range(num_rois):
            if assigned_rois[i]:
                continue
            roi = rois[i]
            obj_idx = int(roi[1])
            if obj_idx == -1:
                continue
            target_obj = pose_rbpf.obj_list[obj_idx]
            add_new_instance = True

            # associate the same object, assume one instance per object
            for j in range(len(pose_rbpf.instance_list)):
                if pose_rbpf.instance_list[j] == target_obj and pose_rbpf.rbpf_ok_list[j] == False:
                    print('initialize previous object: %s' % (target_obj))
                    add_new_instance = False
                    print("here 2")
                    Tco, max_sim = pose_rbpf.pose_estimation_single(j, roi, image_input,
                                                                    image_depth, visualize=visualize)
            if add_new_instance:
                print('initialize new object: %s' % (target_obj))
                pose_rbpf.add_object_instance(target_obj)
                print("here 3")
                Tco, max_sim = pose_rbpf.pose_estimation_single(len(pose_rbpf.instance_list)-1, roi, image_input,
                                                                image_depth, visualize=visualize)


        # save result
        filename = os.path.join(output_dir, f'{video_id}_{image_id}.mat')
        pose_rbpf.save_results_mat(filename)


        # SDF refinement for multiple objects
        if args.refine and image_label is not None:
            # backproject depth
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]
            im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, im_depth)[0]

            index_sdf = []
            for i in range(len(pose_rbpf.instance_list)):
                if pose_rbpf.rbpf_ok_list[i]:
                    index_sdf.append(i)
            if len(index_sdf) > 0:
                pose_rbpf.pose_refine_multiple(sdf_optimizer, posecnn_classes, index_sdf, im_depth,
                    im_pcloud, image_label, steps=50)

        print('=========[%d/%d]==========' % (k, epoch_size))
