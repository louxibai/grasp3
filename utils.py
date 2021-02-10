###################### utils.py ######################
import numpy as np
import math
import pptk


def quat_mul(q1,q2):
    # q = q1*q2
    q = np.zeros(4)
    q[0] = q1[3] * q2[0] - q1[2] * q2[1] + q1[1] * q2[2] + q1[0] * q2[3]
    q[1] = q1[2] * q2[0] + q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3]
    q[2] = -q1[1] * q2[0] + q1[0] * q2[1] + q1[3] * q2[2] + q1[2] * q2[3]
    q[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3]

    return q


def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.

    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def get_pointcloud(depth_img, camera_intrinsics, camera_matrix):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)
    cam_pts_one = np.ones_like(cam_pts_x)
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z, cam_pts_one), axis=1)
    cam_pts_world_t = np.matmul(camera_matrix, np.transpose(cam_pts))
    cam_pts_world = np.transpose(cam_pts_world_t)

    return cam_pts_world


def calculate_distance(x1, y1, z1, x2, y2, z2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist


def remove_clipping(xyz):
    index = []
    for pts in range(0, len(xyz)):
        # calculate x index
        x = xyz[pts][0]
        y = xyz[pts][1]
        z = xyz[pts][2]
        # 0,-0.39098,0.13889
        # -0.5474, 0.0141, 0.7373
        d = calculate_distance(x, y, z, 0.75, 0.0, 0.75)
        if d > 2.0 or d < 0.03:
            index.append(pts)
    xyz = np.delete(xyz, index, axis=0)
    return xyz


def length(v):
  return math.sqrt(dotproduct(v, v))


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def view_pointcloud(filenames):
    cloud = [0, 0, 0]
    cloud = np.reshape(cloud, (1,3))
    for names in filenames:
        xyz = np.load(names)
        print(np.shape(xyz))
        cloud = np.concatenate([cloud, xyz], axis=0)
    v = pptk.viewer(cloud)


def rotate_cloud(point, pose, pcl):
    point = np.reshape(point,(3,1))
    dummy = np.reshape([0, 0, 0, 1],(1,4))
    T = np.concatenate((pose,point),axis=1)
    T_g2w = np.concatenate((T,dummy),axis=0)
    inv = np.linalg.inv(T_g2w)
    T_w2g = inv[:3, :]
    data = np.zeros((len(pcl), 3))
    ones = np.ones((len(pcl), 1))
    pcl_tmp = np.append(pcl, ones, 1)
    data = np.matmul(T_w2g, np.transpose(pcl_tmp))
    # # print(np.shape(pcl_tmp))
    # for i in range(len(pcl)):
    #     data[i] = np.matmul(T_w2g, pcl_tmp[i])
    return np.transpose(data)


# def vox(xyz):
#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(xyz)
#     vox_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, 0.025)
#     voxels = vox_grid.get_voxels()
#     vg = np.zeros((40, 40, 40), dtype=int)
#     for voxel in voxels:
#         voxel_coordinate = voxel.grid_index
#         x = voxel_coordinate[0]
#         y = voxel_coordinate[1]
#         z = voxel_coordinate[2]
#         if x<40 and y<40 and z<40:
#             vg[x,y,z] = 1
#     return vg


def voxelize(xyz, edge_length):
    VOXEL_SIZE = edge_length/32.0
    xyz = np.floor(xyz/VOXEL_SIZE) + 16
    xyz = np.delete(xyz, np.where(xyz > 31)[0], 0)
    xyz = np.delete(xyz, np.where(xyz < 0)[0], 0)
    xyz = xyz.astype(int)
    input_grid = np.zeros((32, 32, 32), dtype=int)
    for p in xyz:
        input_grid[p[0], p[1], p[2]] = 1
    return input_grid
