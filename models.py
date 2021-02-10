###################### models.py ######################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_scatter import scatter_mean
import torch.nn.init as init


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        assert (args.num_layers >= 1), 'Number of layers is not >=1'

        self.conv_output_dim = hidden_dim
        if args.model_type == "GCN":
            self.conv_output_dim = output_dim

        if args.num_layers == 1:
            self.convs.append(conv_model(input_dim, self.conv_output_dim))
        else:
            if args.model_type == "GAT":
                self.convs.append(conv_model(input_dim, hidden_dim, num_heads=8, dropout=args.dropout))
                hidden_dim = 8 * hidden_dim
                self.conv_output_dim = hidden_dim
                for l in range(args.num_layers - 2):
                    self.convs.append(conv_model(hidden_dim, hidden_dim, dropout=args.dropout))
                self.convs.append(conv_model(hidden_dim, self.conv_output_dim, concat=False, dropout=args.dropout))

            else:
                self.convs.append(conv_model(input_dim, hidden_dim))
                for l in range(args.num_layers - 2):
                    self.convs.append(conv_model(hidden_dim, hidden_dim))
                self.convs.append(conv_model(hidden_dim, self.conv_output_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout),
                                     nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.model_type = args.model_type
        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ############################################################################
        x = self.convs[0](x, edge_index)
        if self.num_layers > 1:
            for l in range(1, self.num_layers):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[l](x, edge_index)
        if self.model_type != "GCN":
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.post_mp(x)
        if self.task == "graph":
            #x = scatter_mean(x, batch, dim=0)
            x = pyg_nn.global_max_pool(x, batch)
        ############################################################################

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""

    def __init__(self, in_channels, out_channels, reducer='mean', normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        ############################################################################
        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels, out_channels)
        init.xavier_uniform_(self.lin.weight)
        init.xavier_uniform_(self.agg_lin.weight)
        ############################################################################

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        ############################################################################
        out = F.relu(self.agg_lin(x))
        ############################################################################

        return self.propagate(edge_index, size2=(num_nodes, num_nodes), x=out, ori_x=x)

    def message(self, x_j, edge_index, size2):
        # x_j has shape [E, out_channels]
        return x_j
        row, col = edge_index
        deg = pyg_utils.degree(row, size2[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, ori_x):
        ############################################################################
        agg_x = aggr_out
        ori_x = self.lin(ori_x)
        aggr_out = F.relu(agg_x + ori_x)
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, 2, dim=-1)
        ############################################################################

        return aggr_out


class GAT(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=1, concat=True, dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout

        ############################################################################
        self.lin = nn.Linear(in_channels, out_channels * num_heads)
        ############################################################################

        ############################################################################
        self.att = nn.Parameter(torch.Tensor(num_heads, out_channels * 2))
        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        x = self.lin(x)
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        cat_x = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((cat_x * self.att).sum(-1), 0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)
        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.view(-1, self.heads, self.out_channels).mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GSP3d(nn.Module):
    def __init__(self):
        super(GSP3d, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=0, stride=2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm3d(32),
            # nn.MaxPool3d(2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*5*5*5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            # nn.BatchNorm1d(12),
            nn.Linear(12, 13*13*13),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        shape_embedding = self.cnn1(x1)
        state_embedding = self.fc2(x2)
        batch_size = state_embedding.size()[0]
        state_embedding = torch.reshape(state_embedding, (batch_size, 1, 13, 13, 13))
        state_embedding = state_embedding.repeat((1, 32, 1, 1, 1))
        feature_map = shape_embedding*state_embedding
        x = self.cnn2(feature_map)
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        return output


class CNN3d(nn.Module):
    def __init__(self):
        super(CNN3d, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=0, stride=2),
            nn.ELU(alpha=1.0),
            # nn.BatchNorm3d(32),
            nn.Dropout(p=0.2),

            nn.Conv3d(32, 32, kernel_size=3, padding=0),
            nn.ELU(alpha=1.0),
            # nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Dropout(p=0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6 * 6, 128),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                print('Initialized', m)
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            # elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            #     print('Initialized', m, 'constant')
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x1):
        shape_embedding = self.cnn1(x1)

        # print(np.shape(shape_embedding))

        shape_embedding = shape_embedding.view(shape_embedding.size()[0], -1)
        output = self.fc1(shape_embedding)
        return output


class ReachabilityPredictor(nn.Module):
    def __init__(self):
        super(ReachabilityPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(12, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1):
        output = self.fc1(x1)
        return output


class CarpNetwork(nn.Module):
    def __init__(self):
        super(CarpNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=0, stride=2),
            nn.ELU(alpha=1.0),
            nn.BatchNorm3d(32),
            nn.Dropout(p=0.2),

            nn.Conv3d(32, 32, kernel_size=3, padding=0),
            nn.ELU(alpha=1.0),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Dropout(p=0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            # nn.Linear(192, 128),
            # nn.ReLU(inplace=True),
            nn.Linear(80, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        shape_embedding = self.cnn1(x1)

        state_embedding = self.fc2(x2)
        # print(np.shape(state_embedding))

        shape_embedding = shape_embedding.view(shape_embedding.size()[0], -1)
        shape_embedding = self.fc1(shape_embedding)
        embedding = torch.cat((shape_embedding, state_embedding), dim=1)
        # print(np.shape(embedding))
        output = self.fc3(embedding)
        # output = self.fc1(output)
        return output


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

if __name__ == '__main__':
    from dataset import GNNDataset
    dataset = GNNDataset()
    model = GNNStack
