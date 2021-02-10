import torch
from torch_geometric.data import Data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
from utils import voxelize, rotate_cloud
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import pptk
import matplotlib.pyplot as plt
import sys


def get_files(root):
    class_list = ['imported_part_0', 'imported_part_2', 'imported_part_3', 'imported_part_4', 'imported_part_5',
                  'imported_part_6']
    rgb, depth = [], []
    for object_class in class_list:
        object_dir = os.path.join(root, object_class)
        for filename in os.listdir(object_dir):
            if filename.startswith("rgb"):
                rgb.append(os.path.join(object_dir, filename))
            elif filename.startswith("depth"):
                depth.append(os.path.join(object_dir, filename))
    return rgb, depth


def center_crop_150(rgb_raw):
    hh, ww = np.nonzero(rgb_raw[:, :, 0])
    h_min, h_max = np.min(hh), np.max(hh)
    w_min, w_max = np.min(ww), np.max(ww)
    h_center = int((h_max - h_min) / 2 + h_min)
    w_center = int((w_max - w_min) / 2 + w_min)

    rgb_raw = rgb_raw[h_center - 60:h_center + 60, w_center - 60:w_center + 60, :]
    return rgb_raw


def siamese_data():
    root = 'C:/Users\louxi\Desktop\icra2021\siamese_data'
    x1, x2, y = [], [], []
    rgb_files, depth_files = get_files(root)
    for rgb_file in rgb_files:
        rgb_raw = np.asarray(Image.open(rgb_file))
        rgb_raw = center_crop_150(rgb_raw)
        input_0 = rgb_raw.reshape(3, 150, 150)
        x1.append(input_0)
        # fig = plt.figure(0)
        # sub_1 = fig.add_subplot(1,2,1)
        # sub_1.imshow(rgb_raw)
        label = random.choice([0, 1])
        print(label)
        y.append(label)
        if label == 0:
            current_class = rgb_file[59]
            new_class = random.choice([0, 2, 3, 4, 5, 6])
            while current_class == str(new_class):
                new_class = random.choice([0, 2, 3, 4, 5, 6])
            rgb_file = rgb_file[0:59] + str(new_class) + rgb_file[60:]
            n_rgb_raw = np.asarray(Image.open(rgb_file))
            n_rgb_raw = center_crop_150(n_rgb_raw)
            input_1 = n_rgb_raw.reshape(3, 150, 150)
            x2.append(input_1)
            # sub_2 = fig.add_subplot(1, 2, 2)
            # sub_2.imshow(n_rgb_raw)
            # plt.show()
            # input('s')
        else:
            rgb_file = rgb_file[0:71] + str(random.randint(0, 499)) + '.png'
            p_rgb_raw = np.asarray(Image.open(rgb_file))
            p_rgb_raw = center_crop_150(p_rgb_raw)
            input_1 = p_rgb_raw.reshape(3, 150, 150)
            x2.append(input_1)

            # sub_2 = fig.add_subplot(1, 2, 2)
            # sub_2.imshow(p_rgb_raw)
            # plt.show()
            # input('s')
        print(np.shape(x1), np.shape(x2), np.shape(y))
    print(np.shape(x1), np.shape(x2), np.shape(y))
    np.savez_compressed(root + '/train.npz', x1=x1, x2=x2, y=y)


def siamese_data_rgb():
    root = 'C:/Users\louxi\Desktop\icra2021\siamese_data'
    class_list = ['imported_part_0', 'imported_part_2', 'imported_part_3', 'imported_part_4', 'imported_part_5',
     'imported_part_6']
    input_0, input_1, label = [], [], []
    for object_class in class_list:
        object_dir = os.path.join(root, object_class)
        rgb_files = [filename for filename in os.listdir(object_dir) if filename.startswith("rgb")]
        for idx, rgb_filename in enumerate(rgb_files, 0):
            print(object_class, idx)
            rgb_path = os.path.join(object_dir, rgb_filename)
            rgb_raw = np.asarray(Image.open(rgb_path))
            rgb_raw = center_crop_150(rgb_raw)

            random_rgb = random.choice(rgb_files)
            positive_rgb_path = os.path.join(object_dir, random_rgb)
            positive_rgb_raw = np.asarray(Image.open(positive_rgb_path)) # 150x150x3 ndarray
            positive_rgb_raw = center_crop_150(positive_rgb_raw)

            input_0.append(rgb_raw)
            input_1.append(positive_rgb_raw)
            label.append(1)

            negative_class = random.choice(class_list)
            while negative_class == object_class:
                negative_class = random.choice(class_list)
            negative_rgb_filename = [filename for filename in os.listdir(os.path.join(root, negative_class)) if filename.startswith("rgb")]
            random_negative = random.choice(negative_rgb_filename)
            negative_rgb_path = os.path.join(root, negative_class, random_negative)
            negative_rgb_raw = np.asarray(Image.open(negative_rgb_path)) # 150x150x3 ndarray
            negative_rgb_raw = center_crop_150(negative_rgb_raw)

            input_0.append(rgb_raw)
            input_1.append(negative_rgb_raw)
            label.append(0)
            print(np.shape(input_0), np.shape(input_1), np.shape(label))

    print(np.shape(input_0), np.shape(input_1), np.shape(label))
    # indices = np.arange(input_0.shape[0])
    # np.random.shuffle(indices)
    # input_0 = input_0[indices, :, :, :]
    # input_1 = input_1[indices, :, :, :]
    # label = label[indices]
    np.savez_compressed(root+'/train.npz', input_0=input_0, input_1=input_1, label=label)


def carp_data(root):
    env_cloud = np.load(root+'env.npy')
    poses = [filename for filename in os.listdir(root) if filename.startswith("hand")]
    f = open(os.path.join(root, 'label.txt'))
    label_str = f.readline()
    f.close()
    label = np.fromstring(label_str, dtype=int, sep=',')
    # label = np.ones_like(label) - np.floor(label/100)
    # label = np.load(os.path.join(root, 'balanced_label.npy'))
    # label = label
    print(len(poses), len(label), label, np.sum(label))
    x1 = []
    x2 = []
    for idx in range(len(poses)):
        state = np.load(os.path.join(root, poses[idx]))
        # pptk.viewer(pcd)
        # env_cloud = get_pointcloud(depth_raw, cam_intrinsics, panda.depth_m)
        pt = np.asarray([state[3], state[7], state[11]], dtype=float)
        pose = np.asarray([[state[0], state[1], state[2]],
                           [state[4], state[5], state[6]],
                           [state[8], state[9], state[10]]])
        rotated_cloud = rotate_cloud(pt, pose, env_cloud)
        vg = voxelize(rotated_cloud, 0.2)

        # fig = plt.figure()
        #
        # ax = fig.gca(projection='3d')
        # ax.voxels(vg, facecolors='green', edgecolors='k')
        # plt.show()
        # input('s')
        # xyz = np.asarray([state[3], state[7], state[11]])
        # print(xyz, label[idx])
        x1.append(vg)
        x2.append(state)
        sys.stdout.write("\r Processing {}/{}".format(idx, len(poses)))
        sys.stdout.flush()
    np.savez_compressed(root+'carp_20.npz', x1=x1, x2=x2, y=label)


def zb_data(root):
    env_cloud = np.load(root+'env.npy')
    # poses = [filename for filename in os.listdir(root) if filename.startswith("hand")]
    # objects = [filename for filename in os.listdir(root) if filename.startswith("object")]
    f = open(os.path.join(root, 'label.txt'))
    label_str = f.readline()
    f.close()
    label = np.fromstring(label_str, dtype=int, sep=',')
    # label = np.ones_like(label) - np.floor(label/100)
    # label = np.load(os.path.join(root, 'balanced_label.npy'))
    # label = label
    print(len(label), np.sum(label))
    x1 = []
    x2 = []
    lb = []
    for idx in range(3332):
        state = np.load(os.path.join(root, 'hand_pose_'+str(idx)+'.npy'))
        # print(state)
        # print([state[2], state[6], state[10]])
        # print(label[idx])
        # pcd = np.load(os.path.join(root, objects[idx]))
        # tmp_cloud = np.concatenate((env_cloud, pcd), axis=0)
        # print(np.shape(tmp_cloud))

        tmp_cloud = np.delete(env_cloud, np.where(env_cloud[:, 2] < 0.001), axis=0)
        pt = np.asarray([state[3], state[7], state[11]], dtype=float)
        pose = np.asarray([[state[0], state[1], state[2]],
                           [state[4], state[5], state[6]],
                           [state[8], state[9], state[10]]])
        # print(pose)
        rotated_cloud = rotate_cloud(pt, pose, tmp_cloud)
        vg = voxelize(rotated_cloud, 1.0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(False)
        ax.voxels(vg, facecolors=(1.0, 0.0, 0.0, 1.0), edgecolors='k')
        plt.axis('off')
        plt.savefig('log/voxel_'+str(idx)+'.png')

        x1.append(vg)
        x2.append(state)
        # print(label[idx])
        lb.append(label[idx])
        print(idx, label[idx])

        # sys.stdout.write("\r Processing {}/{}".format(idx, len(poses)))
        # sys.stdout.flush()
    np.savez_compressed(root+'zb.npz', x1=x1, x2=x2, y=lb)


def combine_dataset():
    root = 'C:/Users/louxi/Desktop/icra2021/'
    r1 = 'C:/Users/louxi/Desktop/icra2021/zb/'
    r2 = 'C:/Users/louxi/Desktop/icra2021/zb_2/'
    data_1 = np.load(r1+'zb.npz')
    data_2 = np.load(r2+'zb.npz')
    x1_1 = data_1['x1']
    print(np.shape(x1_1))
    # self.x2 = self.data['x2']
    label_1 = data_1['y']
    print(np.shape(label_1))

    # self.x2 = self.x2.reshape((shape[0], 12, 1))
    x1_2 = data_2['x1']
    print(np.shape(x1_2))

    # self.x2 = self.data['x2']
    label_2 = data_2['y']
    print(np.shape(label_2))

    x1 = np.concatenate((x1_1, x1_2), axis=0)
    y = np.concatenate((label_1, label_2))
    np.savez_compressed(root+'zb.npz', x1=x1, y=y)


def rot2vec(pose):
    abc, _ = cv.Rodrigues(pose)
    np.reshape(abc, (3,))
    theta = np.sqrt(np.sum((abc[0]**2, abc[1]**2, abc[2]**2)))
    x, y, z = abc[0]/theta, abc[1]/theta, abc[2]/theta
    out = [theta, x, y, z]
    return out


def gsp_dataset(root='/home/lou00015/data/gsp', exp_coord=True):
    f = open(os.path.join(root, 'label.txt'))
    label_str = f.readline()
    f.close()
    label = [int(s) for s in label_str]
    # label = np.fromstring(label_str, dtype=int, sep=',')
    print(len(label), np.sum(label))
    x1, x2, y = [], [], []
    for i in range(4000):
        cloud = np.load(root+'/cloud_{}.npy'.format(str(i)))
        state = np.load(root+'/action_{}.npy'.format(str(i)))
        if exp_coord:
            pt = [state[3], state[7], state[11]]
            pose = np.array([[state[0], state[1], state[2]],
                             [state[4], state[5], state[6]],
                             [state[8], state[9], state[10]]])
            txyz = rot2vec(pose)
            action = np.concatenate((txyz, pt))
            cloud = cloud - pt
            v = voxelize(cloud, 0.1)

            x1.append(v)
            x2.append(action)
            y.append(label[i])
    np.savez_compressed(root+'/gsp_train.npz', x1=x1, x2=x2, y=y)


def gnn_dataset(root='/home/lou00015/data/gnn'):
    # cnn3d = torch.load('gsp_gnn.pt')
    f = open(os.path.join(root, 'label.txt'))
    label_str = f.readline()
    f.close()
    label = [int(s) for s in label_str]
    train_data = []
    for graph_i in range(200):
        # cloud = np.load(root+'/cloud_{}.npy'.format(str(graph_i)))
        x, y, edge_index = [], [], None
        for grasp_i in range(25):
            node_ft1 = np.zeros((128, 1), dtype=float)
            node_ft2 = np.load(root+'/cloud_{}_action_{}.npy'.format(str(graph_i), str(grasp_i)))
            node_ft2 = node_ft2.reshape(12, 1)
            x = np.vstack((node_ft1, node_ft2))
            print(np.shape(x))
            y = label[graph_i*25+grasp_i]
        graph = Data(x=x, edge_index=edge_index, y=y)
        train_data.append(graph)
    torch.save(train_data, 'graph_train.pt')


if __name__ == '__main__':
    gsp_dataset()
