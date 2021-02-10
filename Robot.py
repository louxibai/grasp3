import numpy as np
from simulation import sim as vrep
from utils import remove_clipping, rotate_cloud, get_pointcloud, voxelize
import pptk
import time


class Robot(object):
    def __init__(self, clientID):
        self.cid = clientID
        self.dummybyte = bytearray()
        sim_ret, cam_depth_handle = vrep.simxGetObjectHandle(self.cid, 'kinect_depth', vrep.simx_opmode_blocking)
        self.depth_handle = cam_depth_handle
        sim_ret, cam_rgb_handle = vrep.simxGetObjectHandle(self.cid, 'kinect_rgb', vrep.simx_opmode_blocking)
        self.rgb_handle = cam_rgb_handle
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.cid, 'kinect_depth',
                                                                                     vrep.sim_scripttype_childscript,
                                                                                     'getMatrix', [], [], [], self.dummybyte,
                                                                                     vrep.simx_opmode_blocking)
        self.depth_m = np.asarray([[retFloats[0], retFloats[1], retFloats[2], retFloats[3]],
                                   [retFloats[4], retFloats[5], retFloats[6], retFloats[7]],
                                   [retFloats[8], retFloats[9], retFloats[10], retFloats[11]]])

    def get_rgbd_image(self):
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.cid, self.rgb_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.cid, self.depth_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 3
        depth_img = depth_img * (zFar - zNear) + zNear
        return color_img, depth_img

    def get_pointcloud(self):
        result, state, data = vrep.simxReadVisionSensor(self.cid, self.depth_handle, vrep.simx_opmode_blocking)
        data = data[1]
        pcl = []
        for i in range(2, len(data), 4):
            p = [data[i], data[i + 1], data[i + 2], 1]
            pcl.append(np.matmul(self.depth_m, p))
        cloud = remove_clipping(pcl)
        return cloud

    # def get_object_pointcloud(self):
    #     result, state, data = vrep.simxReadVisionSensor(self.cid, self.object_depth_handle, vrep.simx_opmode_blocking)
    #     data = data[1]
    #     pcl = []
    #     for i in range(2, len(data), 4):
    #         p = [data[i], data[i + 1], data[i + 2], 1]
    #         pcl.append(np.matmul(self.depth_m, p))
    #     cloud = remove_clipping(pcl)
    #     return cloud

    def wait(self):
        running = True
        while running:
            res, signal = vrep.simxGetIntegerSignal(self.cid, "finish", vrep.simx_opmode_oneshot_wait)
            if signal == 18:
                running = False
            else:
                running = True


if __name__ == '__main__':
    cid = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if cid != -1:
        vrep.simxStartSimulation(cid, vrep.simx_opmode_blocking)
        panda = Robot(cid)
        cloud = panda.get_pointcloud()
        v = pptk.viewer(cloud)
