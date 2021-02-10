import torch
from sim_env import Robot, scenario_one, add_object, scenario_two
from simulation import sim as vrep
import numpy as np
from models import get_model_instance_segmentation, CarpNetwork, CNN3d
from siamese import SiameseNetwork
from sample import grasp_pose_generation, generate_pose_set
from utils import rotate_cloud, get_pointcloud, voxelize, remove_clipping
import os
import time
import random
import pptk
import matplotlib.pyplot as plt
from dataset import center_crop_150
from PIL import Image
from visualization import vis_grasp

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def pad(img):
    ww = 1280
    hh = 960
    if len(img.shape) == 3:
        ht, wd, cc = img.shape

        # create new image of desired size and color (blue) for padding
        color = (0, 0, 0)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd] = img
        return result
    else:
        ht, wd = img.shape

        # create new image of desired size and color (blue) for padding
        result = np.full((hh, ww), 0, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd] = img
        return result


class Perception(object):
    def __init__(self):
        # input rgbd image in numpy array format [w h c]
        self.sdmrcnn_model = get_model_instance_segmentation(2).to(device, dtype=torch.float)
        self.sdmrcnn_model.load_state_dict(torch.load(os.path.join('19.pth')))
        self.sdmrcnn_model.eval()
        self.siamese_model = SiameseNetwork().cuda()
        self.siamese_model.load_state_dict(torch.load('siamese.pt'))
        self.siamese_model.eval()

    def segmentation(self, raw_rgb, raw_depth):
        rgb_raw_img = np.zeros_like(raw_rgb)
        for i in range(raw_rgb.shape[2]):
            rgb_raw_img[:, :, i] = raw_rgb[:, :, 2 - i]

        color_img = rgb_raw_img.astype(np.float) / 255.
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        for c in range(color_img.shape[2]):
            color_img[:, :, c] = (color_img[:, :, c] - img_mean[c]) / img_std[c]

        depth_img = raw_depth.astype(np.float)
        x, y = np.shape(depth_img)
        depth_img.shape = (x, y, 1)
        # depth_img = depth_img / np.amax(depth_img)
        img = np.concatenate((color_img[:, :, 0:2], depth_img),
                             axis=2)

        test_input = [torch.from_numpy(np.transpose(img, [2, 0, 1])).to(device, dtype=torch.float)]

        output = self.sdmrcnn_model(test_input)
        mask_list = output[0]['masks'].cpu().detach().numpy()
        masks = np.reshape(mask_list, (len(mask_list), 480, 640))
        # masks = []
        # for mask in mask_list:
        #     mask = mask.reshape(960, 1280)
        #     mask = mask[240:720, 320:960]
        #     masks.append(mask)
        return masks

    def classification(self, masks, raw_rgb, anchor_img):
        scores = []
        img0 = anchor_img
        img0 = torch.from_numpy(np.reshape(img0, (1, 3, 120, 120)))
        masks = np.reshape(masks, (len(masks), 480, 640))
        print('Number of objects detected: %d' % len(masks))
        for mask in masks:
            color_img = np.copy(raw_rgb)
            color_img[np.where(mask < 0.5)] = 0
            color_img = pad(color_img)
            color_img = center_crop_150(color_img)

            img1 = np.reshape(color_img, (1, 3, 120, 120))

            img1 = torch.from_numpy(img1)
            img0, img1 = img0.type(torch.FloatTensor), img1.type(torch.FloatTensor)
            img0, img1, = img0.cuda(), img1.cuda()
            output = self.siamese_model(img0, img1)
            output = output.detach().cpu().numpy()
            scores.append(output[0])
        scores = np.asarray(scores)
        res_mask = masks[np.argmax(scores)]
        res_mask = np.reshape(res_mask, (480, 640))
        return res_mask


class Manipulation(object):
    def __init__(self):
        self.cnn3d = CNN3d().cuda()
        self.cnn3d.load_state_dict(torch.load('grasping.pt'))
        self.cnn3d.eval()
        self.carp_model = CNN3d().cuda()
        self.carp_model.load_state_dict(torch.load('collision_1012.pt'))
        self.carp_model.eval()
        # env_cloud = np.load(root + 'env.npy')

    def carp(self, cloud, pt, pose_set):
        scores, collision_free_pose = [], []
        for i in range(len(pose_set)):
            pose = pose_set[i]
            tmp_cloud = rotate_cloud(pt, pose, cloud)
            env_vox = voxelize(tmp_cloud, 1.0)
            env_vox = np.asarray(env_vox, dtype=float)
            env_vox = torch.tensor(env_vox.reshape((1, 1, 32, 32, 32)))
            env_vox = env_vox.type(torch.FloatTensor)
            env_vox = env_vox.cuda()
            yhat = self.carp_model(env_vox)
            yhat = yhat.detach().cpu().numpy()
            scores.append(yhat)
        scores = np.asarray(scores)
        scores = scores.reshape((200,))
        for idx in list(np.where(scores>0.8)[0]):
            collision_free_pose.append(pose_set[idx])
        return np.asarray(collision_free_pose)

    def grasping(self, cloud, pt, pose_set):
        vg_set = []
        for i, pose in enumerate(pose_set, 0):
            point = pt
            r_cloud = rotate_cloud(point, pose, cloud)
            vg = voxelize(r_cloud ,0.1)
            vg_set.append(vg)
        vg_set = np.reshape(vg_set, (len(vg_set), 1, 32, 32, 32))
        vg_set = torch.from_numpy(vg_set)
        vg_set = vg_set.type(torch.FloatTensor)
        vg_set = vg_set.cuda()
        grasping_scores = self.cnn3d(vg_set)
        grasping_scores = grasping_scores.detach().cpu().numpy()
        idx = np.argmax(grasping_scores)
        pose = pose_set[idx]
        return pose


def simulation_experiment_standard():
    cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    object_inventory = ['imported_part_0','imported_part_5','imported_part_2','imported_part_3','imported_part_6', 'imported_part_9']
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    root = 'C:/Users/louxi/Desktop/icra2021/zb_2/'
    p = Perception()  # perception module p
    g = Manipulation()
    # positions = [[0.20, 0, 0.15], [-0.20, 0, 0.15], [0, -0.20, 0.15], [0, 0.20, 0.15]]
    env_cloud = np.load(root + 'env.npy')
    emptyBuff = bytearray()
    if cid != -1:
        i = 0
        planning = 0
        vrep.simxStartSimulation(cid, operationMode=vrep.simx_opmode_blocking)
        panda = Robot(cid)
        # for i in range(10):
            # add_object(cid, 'imported_part_' + str(i), random.choice(positions))
        # for i in range(10):
        #     add_object(cid, 'imported_part_' + str(i), [random.uniform(-0.10, 0.10), random.uniform(-0.10, 0.10), 0.15])
        while True:
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing0', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_0.npy', np.array(retFloats))
            # time.sleep(1)
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing1', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_1.npy', np.array(retFloats))
            #
            # time.sleep(1)
            #
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing2', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_2.npy', np.array(retFloats))
            #
            # time.sleep(1)
            #
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing3', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_3.npy', np.array(retFloats))
            #
            # time.sleep(1)
            #
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing4', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_4.npy', np.array(retFloats))
            #
            # time.sleep(1)
            #
            # res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing5', vrep.sim_scripttype_childscript, 'getlanding', [],
            #                             [], [], emptyBuff, vrep.simx_opmode_blocking)
            # print(retFloats)
            # np.save('pose_5.npy', np.array(retFloats))
            #
            # time.sleep(1)
            #
            # input('s')

            # objects_nb = input('number')
            objects_nb = '0'
            rgb_raw, depth_raw = panda.get_rgbd_image()
            # im = Image.fromarray(rgb_raw)
            # im.save('imported_part_5.png')
            # input('save?')
            # Perception
            m_list = p.segmentation(rgb_raw, depth_raw)  # segment scene
            anchor = np.asarray(Image.open('C:/Users\louxi\Desktop\icra2021\siamese_data\inventory/' + object_inventory[int(objects_nb)] + '.png'))
            anchor = center_crop_150(anchor)
            object_mask = p.classification(m_list, rgb_raw, anchor)  # load anchor here

            # object_depth[np.where(object_mask < 0.5)] = 0
            rgb_raw[np.where(object_mask < 0.5)] = 0
            # plt.imshow(rgb_raw)
            # plt.show()
            object_depth = np.copy(depth_raw)
            object_depth[np.where(object_mask < 0.5)] = 0
            object_cloud = get_pointcloud(object_depth, cam_intrinsics, panda.depth_m)
            ws = [-0.25, 0.25, -0.25, 0.25, 0.001, 0.10]
            segmented_cloud = np.array([pts for pts in object_cloud if pts[0] < ws[1] and pts[0] > ws[0] and pts[1] < ws[3] and pts[1] > ws[2] and pts[2] < ws[5] and pts[2] > ws[4]])
            # pptk.viewer(segmented_cloud)
            recognized = 'yes'
            if recognized == 'yes':
                initial_poses, dummy_points = grasp_pose_generation(60, segmented_cloud, 200)
                # collision_free_poses = g.carp(env_cloud, np.average(segmented_cloud, axis=0), initial_poses)
                # print(len(collision_free_poses))
                pt = np.average(segmented_cloud, axis=0)
                print('landing'+str(i))
                pose = np.load('pose_'+str(i)+'.npy')
                i = i + 1
                # landing_mtx = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                #                pose[1][0], pose[1][1], pose[1][2], pt[1],
                #                pose[2][0], pose[2][1], pose[2][2], pt[2]]
                landing_mtx = pose
                print(landing_mtx)
                # vis_grasp(pose, pt, segmented_cloud)
                ending_mtx = np.copy(landing_mtx)
                ending_mtx[-1] = ending_mtx[-1] + 0.15
                # recognized = input('correct?')
                if recognized == 'yes':
                    vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [],
                                                landing_mtx, [],
                                                emptyBuff, vrep.simx_opmode_blocking)
                    vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [],
                                                ending_mtx, [],
                                                emptyBuff, vrep.simx_opmode_blocking)
                    time.sleep(0.5)
                    vrep.simxCallScriptFunction(cid, 'Sphere', vrep.sim_scripttype_childscript, 'grasp', [], [], [],
                                                emptyBuff,
                                                vrep.simx_opmode_blocking)

                    while True:
                        res, finish = vrep.simxGetIntegerSignal(cid, "finish", vrep.simx_opmode_blocking)
                        if finish == 18:
                            res, result = vrep.simxGetIntegerSignal(cid, "collision", vrep.simx_opmode_blocking)
                            if result == 1:
                                planning = planning + 1
                            print('result is %d. start next experiment' % result)
                            break

                    time.sleep(3.0)
            else:
                continue


def simulation_experiment_challenging():
    cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    object_inventory = ['imported_part_0', 'imported_part_2', 'imported_part_3', 'imported_part_6',
                        'imported_part_7',
                        'imported_part_8']
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    root = 'C:/Users/louxi/Desktop/icra2021/zb_2/'
    p = Perception()  # perception module p
    g = Manipulation()
    positions = [[0.20, 0, 0.15], [-0.20, 0, 0.15], [0, -0.20, 0.15], [0, 0.20, 0.15]]
    env_cloud = np.load(root + 'env.npy')
    if cid != -1:
        i = 0
        planning = 0
        vrep.simxStartSimulation(cid, operationMode=vrep.simx_opmode_blocking)
        panda = Robot(cid)
        for i in range(10):
            add_object(cid, 'imported_part_' + str(i), random.choice(positions))
        while i < 51:
            print('experiment number: ', i)
            rgb_raw, depth_raw = panda.get_rgbd_image()
            # Perception
            m_list = p.segmentation(rgb_raw, depth_raw)  # segment scene
            target_object = random.choice(object_inventory)
            print(target_object)
            anchor = np.asarray(Image.open(os.path.join('C:/Users\louxi\Desktop\icra2021\siamese_data', target_object,
                                                        'rgb_image_' + str(random.randint(0, 499)) + '.png')))
            anchor = center_crop_150(anchor)
            object_mask = p.classification(m_list, rgb_raw, anchor)  # load anchor here

            # object_depth[np.where(object_mask < 0.5)] = 0
            rgb_raw[np.where(object_mask < 0.5)] = 0
            object_depth = np.copy(depth_raw)
            object_depth[np.where(object_mask < 0.5)] = 0
            object_cloud = get_pointcloud(object_depth, cam_intrinsics, panda.depth_m)
            ws = [-0.25, 0.25, -0.25, 0.25, 0.001, 0.10]
            segmented_cloud = np.array([pts for pts in object_cloud if
                                        pts[0] < ws[1] and pts[0] > ws[0] and pts[1] < ws[3] and pts[1] > ws[
                                            2] and
                                        pts[2] < ws[5] and pts[2] > ws[4]])
            if len(segmented_cloud) == 0:
                continue
            # pptk.viewer(segmented_cloud)
            #
            # fig = plt.figure(0)
            # sub_1 = fig.add_subplot(1, 2, 1)
            # sub_1.imshow(anchor)
            # sub_2 = fig.add_subplot(1, 2, 2)
            # sub_2.imshow(rgb_raw)
            # plt.show()

            initial_poses, dummy_points = grasp_pose_generation(90, segmented_cloud, 200)
            # collision_free_poses = g.carp(env_cloud, np.average(segmented_cloud, axis=0), initial_poses)
            landing_mtx = g.grasping(segmented_cloud, np.average(segmented_cloud, axis=0), initial_poses)
            ending_mtx = np.copy(landing_mtx)
            ending_mtx[-1] = ending_mtx[-1] + 0.15
            emptyBuff = bytearray()
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing0', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing1', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing2', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing3', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing4', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(cid, 'landing5', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            print(retFloats)
            input('s')
            vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [],
                                        landing_mtx, [],
                                        emptyBuff, vrep.simx_opmode_blocking)
            vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [],
                                        ending_mtx, [],
                                        emptyBuff, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            vrep.simxCallScriptFunction(cid, 'Sphere', vrep.sim_scripttype_childscript, 'grasp', [], [], [],
                                        emptyBuff,
                                        vrep.simx_opmode_blocking)

            while True:
                res, finish = vrep.simxGetIntegerSignal(cid, "finish", vrep.simx_opmode_blocking)
                if finish == 18:
                    res, result = vrep.simxGetIntegerSignal(cid, "collision", vrep.simx_opmode_blocking)
                    if result == 1:
                        planning = planning + 1
                    print('result is %d. start next experiment' % result)
                    break

            i = i + 1
            # vrep.simxStopSimulation(cid, operationMode=vrep.simx_opmode_blocking)
            time.sleep(3.0)
            # input('s')
        print(planning)


if __name__ == '__main__':
    simulation_experiment_standard()

