import math
from math import cos, sin, pi, sqrt
import random
import numpy as np


def push_pose_generation(pcd, nb):
    push_list = []
    for j in range(nb):
        pt = random.choice(pcd)
        pt = [pt[0],pt[1],pt[2]+0.03]
        gamma = float(random.randint(0,628))/100.0
        angle = [-3.14, 0, gamma]
        landing = [pt[0]-0.1*math.sin(gamma), pt[1]-0.1*math.cos(gamma), pt[2]]
        ending = [pt[0]+0.1*math.sin(gamma), pt[1]+0.1*math.cos(gamma), pt[2]]
        pose = [pt, angle, landing, ending]
        push_list.append(pose)
    return push_list


def push_transform(push_pose, pcl):
    push_point = push_pose[0]
    push_angle = push_pose[1]
    a,b,g = push_angle[0],push_angle[1],push_angle[2]
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a), np.cos(a)]], dtype=np.float32)
    Ry = np.array([[np.cos(b), 0.0,  np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]], dtype=np.float32)
    Rz = np.array([[np.cos(g), -np.sin(g), 0.0],
                   [np.sin(g), np.cos(g), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.dot(Rz, np.dot(Ry, Rx))
    translation = np.reshape(push_point,(3,1))
    dummy = np.asarray([0, 0, 0, 1])
    h_g = np.reshape(dummy,(1,4))
    R_upper = np.concatenate((R,translation),axis=1)
    T_g2w = np.concatenate((R_upper,h_g),axis=0)
    inv = np.linalg.inv(T_g2w)
    T_w2g = inv[:3, :]
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    for i in range(len(pcl)):
        data[i] = np.matmul(T_w2g, pcl_tmp[i])
    return data


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def length(v):
  return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def generate_rand_pose(threshold, low=1):
    # neg_z = [0,0,-1]
    # gamma = np.random.triangular(0, 0, threshold, 1)
    if threshold == 0:
        pose = generate_rand_top_pose()
    else:
        gamma = np.random.randint(low,threshold)
        gamma = float(abs(gamma))
        ap_z = -math.cos(gamma*0.0174533)
        ap_x = random.uniform(-math.sqrt(1-ap_z*ap_z),math.sqrt(1-ap_z*ap_z))
        ap_y = random.choice([math.sqrt(1-ap_x*ap_x-ap_z*ap_z), -math.sqrt(1-ap_x*ap_x-ap_z*ap_z)])
        approach = np.array([ap_x, ap_y, ap_z])
        # cal_gamma = angle(approach, neg_z)
        # print(np.true_divide(cal_gamma,0.0175))

        base0 = np.array([ap_y, -ap_x, 0])
        b0_norm = base0/np.linalg.norm(base0)
        base1 = np.cross(approach,base0)
        b1_norm = base1/np.linalg.norm(base1)
        rand_theta = random.uniform(0, 2*np.pi)
        b0 = np.cos(rand_theta)*b0_norm
        b1 = np.sin(rand_theta)*b1_norm
        binormal = b0 + b1


        axis = np.cross(approach, binormal)
        approach = np.reshape(approach,(3,1))
        binormal = np.reshape(binormal,(3,1))
        axis = np.reshape(axis,(3,1))
        pose = np.concatenate((binormal,axis,approach), axis=1)
        pose[np.isnan(pose)] = 0
    return pose


def generate_pose_set():
    pose_set = []
    betas = [0, 15 * 0.0174533, 30 * 0.0174533]
    for beta in betas:
        if beta == 0:
            for i in range(1, 5):
                z = np.array([0, 0, -1], dtype=float)
                theta = i * np.pi / 4
                x = np.array([np.cos(theta), np.sin(theta), 0], dtype=float)
                y = np.cross(z, x)
                x = np.reshape(x, (3, 1))
                y = np.reshape(y, (3, 1))
                z = np.reshape(z, (3, 1))
                pose = np.concatenate((x, y, z), axis=1)
                pose = np.around(pose, decimals=4)
                pose_set.append(pose)
        else:
            for i in range(8):
                alpha = i * np.pi / 4
                z_z = -cos(beta)
                z_x = sqrt(1-z_z*z_z)*sin(alpha)
                z_y = sqrt(1 - z_z * z_z)*cos(alpha)
                z = np.array([z_x, z_y, z_z])
                base0 = np.array([z_y, -z_x, 0])
                b0_norm = base0 / np.linalg.norm(base0)
                base1 = np.cross(z, base0)
                b1_norm = base1 / np.linalg.norm(base1)
                for j in range(0, 4):
                    theta = j*pi/4
                    b0 = np.cos(theta) * b0_norm
                    b1 = np.sin(theta) * b1_norm
                    x = b0 + b1
                    y = np.cross(z, x)
                    zz = np.reshape(z, (3, 1))
                    yy = np.reshape(y, (3, 1))
                    xx = np.reshape(x, (3, 1))
                    pose = np.concatenate((xx, yy, zz), axis=1)
                    pose = np.around(pose, decimals=4)
                    pose_set.append(pose)
    return pose_set





def generate_rand_top_pose():
    approach = [0, 0, -1]
    bi_x = random.uniform(-100,100)
    bi_y = random.uniform(-100,100)
    bi_z = 0
    binormal = bi_x, bi_y, bi_z
    bi_d = math.sqrt(bi_x*bi_x+bi_y*bi_y+bi_z*bi_z)
    binormal = np.true_divide(binormal,bi_d)
    axis = np.cross(approach, binormal)
    approach = np.reshape(approach,(3,1))
    binormal = np.reshape(binormal,(3,1))
    axis = np.reshape(axis,(3,1))
    pose = np.concatenate((binormal,axis,approach), axis=1)
    return pose


def generate_rand_top_pose_d(i):
    z = np.array([0, 0, -1], dtype=float)
    theta = i*np.pi/4
    x = np.array([np.cos(theta), np.sin(theta), 0], dtype=float)
    y = np.cross(z, x)
    # y = np.array([-np.sin(theta), np.cos(theta), 0], dtype=float)
    print(np.round(theta/0.0174533))
    # bi_x = random.uniform(-100,100)
    # bi_y = random.uniform(-100,100)
    # bi_z = 0
    # binormal = bi_x, bi_y, bi_z
    # bi_d = math.sqrt(bi_x*bi_x+bi_y*bi_y+bi_z*bi_z)
    # binormal = np.true_divide(binormal,bi_d)
    # axis = np.cross(approach, binormal)
    x = np.reshape(x,(3,1))
    y = np.reshape(y,(3,1))
    z = np.reshape(z,(3,1))
    pose = np.concatenate((x, y, z), axis=1)
    print(np.around(pose, decimals=4))
    return np.around(pose, decimals=4)


def grasp_pose_generation(degree,pcd,nb):
    pose = []
    point = []
    sample_cloud = np.delete(pcd, np.where(pcd[:, 2] < 0.015), axis=0)
    for j in range(nb):
        pt = random.choice(sample_cloud)
        point.append(pt)
        xyz = generate_rand_pose(degree, low=1)
        pose.append(xyz)
    return pose, point


def top_grasp_pose(pcd,nb):
    pose = []
    point = []
    for j in range(nb):
        pt = random.choice(pcd)
        point.append(pt)
        xyz = generate_rand_top_pose()
        pose.append(xyz)
    return pose, point


def random_pushing(pcd):
    pse = generate_rand_top_pose()
    pt = random.choice(pcd)
    center = np.average(pcd, axis=0)
    x = center-pt
    x[2] = 0
    # z = [0, 0, -1]
    # bi_x = random.uniform(-100,100)
    # bi_y = random.uniform(-100,100)
    # bi_z = 0
    # binormal = bi_x, bi_y, bi_z
    # bi_d = math.sqrt(bi_x*bi_x+bi_y*bi_y+bi_z*bi_z)
    # binormal = np.true_divide(binormal,bi_d)
    # axis = np.cross(z, binormal)
    # approach = np.reshape(z,(3,1))
    # binormal = np.reshape(binormal,(3,1))
    # axis = np.reshape(axis,(3,1))
    # pose = np.concatenate((binormal,axis,approach), axis=1)

    pt[2] = 0.02
    l = 0.02
    direction = -l*pse[:,0]
    direction[2] = 0
    # dz = np.random.choice([-np.sqrt(l**2-dx**2-dy**2),np.sqrt(l**2-dx**2-dy**2)])
    # direction = [dx, dy, 0]

    landing = [pt[0]-2.5*direction[0], pt[1]-2.5*direction[1], pt[2]]
    ending = [pt[0]+direction[0], pt[1]+direction[1], pt[2]]

    return pt, pse, direction, landing, ending


def random_pushing_enhanced(pcd):
    center = np.average(pcd, axis=0)
    pt = random.choice(pcd)
    x = center-pt
    x[2] = 0
    x = x/np.linalg.norm(x)

    z = [0, 0, -1]
    y = np.cross(z, x)
    y = np.reshape(y,(3,1))
    z = np.reshape(z,(3,1))
    x = np.reshape(x,(3,1))
    pose = np.concatenate((x,y,z), axis=1)
    print(pose)

    pt[2] = 0.02
    l = 0.01
    direction = -l*pose[:,0]
    direction[2] = 0
    # dz = np.random.choice([-np.sqrt(l**2-dx**2-dy**2),np.sqrt(l**2-dx**2-dy**2)])
    # direction = [dx, dy, 0]

    landing = [pt[0]+2.5*direction[0], pt[1]+2.5*direction[1], pt[2]]
    ending = [pt[0]-direction[0], pt[1]-direction[1], pt[2]]

    return pt, pose, direction, landing, ending


def discretized_pushing(pcd):
    pt = np.average(pcd, axis=0)
    dir = np.random.randint(0,8)
    d = []
    l = 0.02
    if dir == 0:
        d = [l, 0, 0]
    if dir == 1:
        d = [np.sqrt(l**2/2), np.sqrt(l**2/2), 0]
    if dir == 2:
        d = [0, l, 0]
    if dir == 3:
        d = [-np.sqrt(l**2/2), np.sqrt(l**2/2), 0]
    if dir == 4:
        d = [-l, 0, 0]
    if dir == 5:
        d = [-np.sqrt(l**2/2), -np.sqrt(l**2/2), 0]
    if dir == 6:
        d = [0, -l, 0]
    if dir == 7:
        d = [np.sqrt(l**2/2), -np.sqrt(l**2/2), 0]

    landing = [pt[0]-3*d[0], pt[1]-3*d[1], pt[2]-0.01]
    ending = [pt[0]+d[0], pt[1]+d[1], pt[2]-0.01]
    return pt, d, landing, ending


def random_action(pcd):
    centroid = np.average(pcd, axis=0)
    x = random.uniform(centroid[0]-0.04, centroid[0]+0.04)
    y = random.uniform(centroid[1]-0.04, centroid[1]+0.04)
    z = random.uniform(0.015, 0.04)
    target = [x, y, z]
    print(target)

    action = [0, 0, 0]
    i = random.sample([0, 1, 2], 1)[0]
    l = random.sample([-0.02, 0.02], 1)[0]
    if i==2:
        action[i] = 0.02
    else:
        action[i] = l
    print(action)

    pose = generate_rand_top_pose()
    return target, pose, action



if __name__ == '__main__':
    generate_pose_set()


