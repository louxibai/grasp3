import torch
from models import GSP3d
import time
import numpy as np
from simulation import sim as vrep
import random
import pptk
from sample import grasp_pose_generation, generate_pose_set
from Robot import Robot
from utils import voxelize, rotate_cloud

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def add_object(cid, object_name, object_pos):
    # object_name = random.choice(['apple', 'banana', 'sugar_box', 'cracker_box', 'mustard_bottle', 'lemon', 'orange',
    #                              'tomato_soup_can'])
    res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
    # object_pos = [random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.20]
    object_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
    # object_angle = [0, 0, 0]
    vrep.simxSetObjectOrientation(cid, object_handle, -1, object_angle, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectPosition(cid, object_handle, -1, object_pos, vrep.simx_opmode_oneshot)
    time.sleep(1.0)
    return object_name, object_handle


def gsp_test():
    wd = '/home/lou00015/data/gsp_test/'
    model = GSP3d().cuda()
    model.load_state_dict(torch.load('gsp.pt'))
    model.eval()
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    eid = 0
    nb_grasp = 300
    if cid != -1:
        pos = [0, 0, 0.15]
        while True:
            vrep.simxStartSimulation(cid, vrep.simx_opmode_blocking)
            panda = Robot(cid)
            obj_name, obj_hdl = add_object(cid, 'imported_part_0', pos)
            time.sleep(1.0)
            cloud = panda.get_pointcloud()
            centroid = np.average(cloud, axis=0)
            if len(cloud) == 0:
                print('no cloud found')
                continue
            elif centroid[2] > 0.045:
                print('perception error')
                continue
            # np.save(wd + 'cloud_' + str(eid) + '.npy', cloud) # save point cloud
            cloud = np.delete(cloud, np.where(cloud[:,2]<0.015), axis=0)
            v = voxelize(cloud-centroid, 0.1)
            pose_set, pt_set = grasp_pose_generation(45, cloud, nb_grasp)
            x1, x2 = [], []
            emptyBuff = bytearray()
            for i in range(nb_grasp):
                pose = pose_set[i]
                pt = pt_set[i]
                landing_mtx = np.asarray([pose[0][0],pose[0][1],pose[0][2],pt[0],
                               pose[1][0],pose[1][1], pose[1][2],pt[1],
                               pose[2][0],pose[2][1],pose[2][2],pt[2]])
                x1.append(v)
                x2.append(landing_mtx)
            x1, x2 = np.stack(x1), np.stack(x2)
            X1 = torch.tensor(x1.reshape((x2.shape[0], 1, 32, 32, 32)), dtype=torch.float, device=device)
            X2 = torch.tensor(x2.reshape((x2.shape[0], 12)), dtype=torch.float, device=device)
            yhat = model(X1, X2)
            yhat = yhat.detach().cpu().numpy()
            scores = np.asarray(yhat)
            scores = scores.reshape((nb_grasp,))
            g_index = np.argmax(scores)
            print('Highest score: {}, the {}th.'.format(str(scores[g_index]), str(g_index)))
            pose = pose_set[g_index]
            pt = centroid
            landing_mtx = np.asarray([pose[0][0], pose[0][1], pose[0][2], pt[0],
                                      pose[1][0], pose[1][1], pose[1][2], pt[1],
                                      pose[2][0], pose[2][1], pose[2][2], pt[2]])
            vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [], landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            ending_mtx = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                              pose[1][0], pose[1][1], pose[1][2], pt[1],
                              pose[2][0], pose[2][1], pose[2][2], pt[2]+0.15]
            vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [], ending_mtx,
                                        [], emptyBuff, vrep.simx_opmode_blocking)
            time.sleep(1.0)
            print('executing experiment %d: ' % g_index)
            print('at: ', pt)
            vrep.simxCallScriptFunction(cid, 'Sphere', vrep.sim_scripttype_childscript, 'grasp', [], [], [], emptyBuff, vrep.simx_opmode_blocking)
            while True:
                res, finish = vrep.simxGetIntegerSignal(cid, "finish", vrep.simx_opmode_oneshot_wait)
                if finish == 18:
                    res, end_pos = vrep.simxGetObjectPosition(cid, obj_hdl, -1, vrep.simx_opmode_blocking)
                    break
            if end_pos[2]>0.05:
                label = 1
            else:
                label = 0
            print(label)
            # f = open(wd + 'label.txt', 'a+')
            # f.write(str(label))
            # f.close()
            eid += 1
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


def count_success():
    wd = 'C:/Users/louxi/Desktop/collision_data/train/'
    f = open(wd + 'label.txt')
    l = []
    label = f.readline()
    for i in range(9000):
        l.append(int(label[i]))
    print(sum(l))
    print(sum(l)/9000)


if __name__ == '__main__':
    gsp_test()
