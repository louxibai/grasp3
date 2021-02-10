import time
import numpy as np
from simulation import sim as vrep
import random
import pptk
from sample import grasp_pose_generation, generate_pose_set
from Robot import Robot

def scenario_one(cid):
    name_list = []
    locations = [[0.15, 0, 0.1], [-0.15, 0, 0.1], [0, 0.15, 0.15],[0, -0.15, 0.1]]
    handle_list = []
    for i in range(0, 4):
        object_name = 'imported_part_' + str(i)
        res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
        # object_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
        object_angle = [0, 0, 0]
        vrep.simxSetObjectOrientation(cid, object_handle, -1, object_angle, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(cid, object_handle, -1, locations[i], vrep.simx_opmode_oneshot)
        time.sleep(1.0)
        name_list.append(object_name)
        handle_list.append(object_handle)
    return name_list, handle_list


def scenario_two(cid):
    name_list = []
    locations = [[0.15, 0, 0.1], [-0.15, 0, 0.1], [0, 0.15, 0.15],[0, -0.15, 0.1]]
    handle_list = []
    for i in range(0, 1):
        object_name = 'imported_part_' + str(i)
        res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
        # object_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
        object_angle = [0, 0, 0]
        vrep.simxSetObjectOrientation(cid, object_handle, -1, object_angle, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(cid, object_handle, -1, locations[i], vrep.simx_opmode_oneshot)
        time.sleep(1.0)
        name_list.append(object_name)
        handle_list.append(object_handle)
    return name_list, handle_list


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


def add_structure(cid):
    object_name = 'box_large'
    res, object_handle = vrep.simxGetObjectHandle(cid, object_name, vrep.simx_opmode_oneshot_wait)
    # object_pos = [random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.20]
    object_pos = [0, 0, 0.075]
    object_angle = [-1.5708, random.uniform(-1.5707, 1.5707), -1.5708]
    vrep.simxSetObjectOrientation(cid, object_handle, -1, object_angle, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectPosition(cid, object_handle, -1, object_pos, vrep.simx_opmode_oneshot)
    time.sleep(1.0)
    return object_name, object_handle


def grasp_gnn_data():
    wd = '/home/lou00015/data/gnn/'
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    eid = 563
    nb_grasp = 25
    if cid != -1:
        pos = [0, 0, 0.15]
        while True:
            vrep.simxStartSimulation(cid, vrep.simx_opmode_blocking)
            panda = Robot(cid)
            obj_name, obj_hdl = add_object(cid, 'imported_part_0', pos)
            time.sleep(1.0)
            cloud = panda.get_pointcloud()
            centroid = np.average(cloud, axis=0)
            print('centroid: ', centroid)
            res, init_pos = vrep.simxGetObjectPosition(cid, obj_hdl, -1, vrep.simx_opmode_blocking)
            res, init_ori = vrep.simxGetObjectOrientation(cid, obj_hdl, -1, vrep.simx_opmode_blocking)
            if len(cloud) == 0:
                print('no cloud found')
                continue
            elif centroid[2] > 0.045:
                print('perception error')
                continue
            np.save(wd + 'cloud_' + str(eid) + '.npy', cloud) # save point cloud
            cloud = np.delete(cloud, np.where(cloud[:,2]<0.015), axis=0)
            pose_set, pt_set = grasp_pose_generation(45, cloud, nb_grasp)
            for i in range(0, nb_grasp):
                pose = pose_set[i]
                pt = pt_set[i]
                emptyBuff = bytearray()
                landing_mtx = [pose[0][0],pose[0][1],pose[0][2],pt[0],
                               pose[1][0],pose[1][1], pose[1][2],pt[1],
                               pose[2][0],pose[2][1],pose[2][2],pt[2]]
                np.save(wd + 'cloud_' + str(eid) + '_action_' + str(i) + '.npy', landing_mtx) # save action
                vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [], landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
                ending_mtx = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                              pose[1][0], pose[1][1], pose[1][2], pt[1],
                              pose[2][0], pose[2][1], pose[2][2], pt[2]+0.15]
                vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [], ending_mtx,
                                            [], emptyBuff, vrep.simx_opmode_blocking)
                time.sleep(1.0)
                print('executing experiment %d: ' % (eid*25+i))
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
                f = open(wd + 'label.txt', 'a+')
                f.write(str(label))
                f.close()
                vrep.simxSetObjectPosition(cid, obj_hdl, -1, init_pos, vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(cid, obj_hdl, -1, init_ori, vrep.simx_opmode_blocking)
            eid += 1
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


def grasp_data():
    wd = '/home/lou00015/data/gsp/'
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    eid = 0
    nb_grasp = 25
    if cid != -1:
        pos = [0, 0, 0.15]
        while True:
            vrep.simxStartSimulation(cid, vrep.simx_opmode_blocking)
            panda = Robot(cid)
            obj_name, obj_hdl = add_object(cid, 'imported_part_0', pos)
            time.sleep(1.0)
            cloud = panda.get_pointcloud()
            centroid = np.average(cloud, axis=0)
            print('centroid: ', centroid)
            if len(cloud) == 0:
                print('no cloud found')
                continue
            elif centroid[2] > 0.045:
                print('perception error')
                continue
            np.save(wd + 'cloud_' + str(eid) + '.npy', cloud) # save point cloud
            cloud = np.delete(cloud, np.where(cloud[:,2]<0.015), axis=0)
            pose_set, pt_set = grasp_pose_generation(45, cloud, nb_grasp)
            pose = pose_set[10]
            pt = pt_set[10]
            emptyBuff = bytearray()
            landing_mtx = [pose[0][0],pose[0][1],pose[0][2],pt[0],
                           pose[1][0],pose[1][1], pose[1][2],pt[1],
                           pose[2][0],pose[2][1],pose[2][2],pt[2]]
            np.save(wd + 'action_' + str(eid) + '.npy', landing_mtx) # save action
            vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [], landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            ending_mtx = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                          pose[1][0], pose[1][1], pose[1][2], pt[1],
                          pose[2][0], pose[2][1], pose[2][2], pt[2]+0.15]
            vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [], ending_mtx,
                                        [], emptyBuff, vrep.simx_opmode_blocking)
            time.sleep(1.0)
            print('executing experiment %d: ' % (eid))
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
            f = open(wd + 'label.txt', 'a+')
            f.write(str(label))
            f.close()
            eid += 1
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


def grasp_centroid_data():
    wd = '/home/lou00015/data/gsp/'
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    eid = 0
    nb_grasp = 25
    if cid != -1:
        pos = [0, 0, 0.15]
        while True:
            vrep.simxStartSimulation(cid, vrep.simx_opmode_blocking)
            panda = Robot(cid)
            obj_name, obj_hdl = add_object(cid, 'imported_part_0', pos)
            time.sleep(1.0)
            cloud = panda.get_pointcloud()
            centroid = np.average(cloud, axis=0)
            print('centroid: ', centroid)
            if len(cloud) == 0:
                print('no cloud found')
                continue
            elif centroid[2] > 0.045:
                print('perception error')
                continue
            np.save(wd + 'cloud_' + str(eid) + '.npy', cloud) # save point cloud
            cloud = np.delete(cloud, np.where(cloud[:,2]<0.015), axis=0)
            pose_set, pt_set = grasp_pose_generation(45, cloud, nb_grasp)
            pose = pose_set[10]
            pt = centroid
            emptyBuff = bytearray()
            landing_mtx = [pose[0][0],pose[0][1],pose[0][2],pt[0],
                           pose[1][0],pose[1][1], pose[1][2],pt[1],
                           pose[2][0],pose[2][1],pose[2][2],pt[2]]
            np.save(wd + 'action_' + str(eid) + '.npy', landing_mtx) # save action
            vrep.simxCallScriptFunction(cid, 'landing', vrep.sim_scripttype_childscript, 'setlanding', [], landing_mtx, [], emptyBuff, vrep.simx_opmode_blocking)
            ending_mtx = [pose[0][0], pose[0][1], pose[0][2], pt[0],
                          pose[1][0], pose[1][1], pose[1][2], pt[1],
                          pose[2][0], pose[2][1], pose[2][2], pt[2]+0.15]
            vrep.simxCallScriptFunction(cid, 'ending', vrep.sim_scripttype_childscript, 'setending', [], ending_mtx,
                                        [], emptyBuff, vrep.simx_opmode_blocking)
            time.sleep(1.0)
            print('executing experiment %d: ' % (eid))
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
            f = open(wd + 'label.txt', 'a+')
            f.write(str(label))
            f.close()
            eid += 1
    else:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
    exit()


if __name__ == '__main__':
    grasp_data()
