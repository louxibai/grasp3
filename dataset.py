import torch
from torch_geometric.data import Data
import numpy as np
from torch.utils.data import Dataset, DataLoader



class GNNDataset(Dataset):
    def __init__(self, root='/home/lou00015/data/gsp', transform=None):
        self.data = np.load(root+'/gsp_train.npz')
        self.x1 = self.data['x1']
        shape = np.shape(self.x1)
        self.x2 = self.data['x2']
        self.label = self.data['y']
        self.x1 = self.x1.reshape((shape[0], 1, 32, 32, 32))
        self.x2 = self.x2.reshape((shape[0], 12))
        self.label = self.label.reshape((shape[0], 1))

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        pose = self.x2[idx]
        # xyz = [pose[3], pose[7]]
        # return self.x2[idx], self.label[idx]
        # return self.x1[idx], np.asarray(xyz, dtype=float), self.label[idx]
        return self.x1[idx], self.x2[idx], self.label[idx]


class GSPDataset(Dataset):
    def __init__(self, root='/home/lou00015/data/gsp', transform=None):
        self.data = np.load(root+'/gsp_train.npz')
        self.x1 = self.data['x1']
        shape = np.shape(self.x1)
        self.x2 = self.data['x2']
        self.label = self.data['y']
        self.x1 = self.x1.reshape((shape[0], 1, 32, 32, 32))
        self.x2 = self.x2.reshape((shape[0], 12))
        self.label = self.label.reshape((shape[0], 1))

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        pose = self.x2[idx]
        # xyz = [pose[3], pose[7]]
        # return self.x2[idx], self.label[idx]
        # return self.x1[idx], np.asarray(xyz, dtype=float), self.label[idx]
        return self.x1[idx], self.x2[idx], self.label[idx]
