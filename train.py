import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import GSP3d, CNN3d, ReachabilityPredictor
from dataset import GSPDataset, GNNDataset
import matplotlib.pyplot as plt
import sys
from torch.utils.data.sampler import SubsetRandomSampler


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_gsp(num_epochs):
    dataset = GSPDataset()
    batch_size = 64
    validation_split = .1
    shuffle_dataset = True
    # random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    print('%d data for training, %d data for validation.' % (dataset_size-split, split))
    if shuffle_dataset:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    net = GSP3d().cuda()
    print('Number of parameters: %d' % count_parameters(net))
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    counter = []
    loss_history = []
    iteration_number = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            net.train()
            x1_train, x2_train, label = data
            x1_train, x2_train, label = x1_train.type(torch.FloatTensor), x2_train.type(torch.FloatTensor), label.type(torch.FloatTensor)
            x1_train, x2_train, label = x1_train.cuda(), x2_train.cuda(), label.cuda()
            optimizer.zero_grad()

            y_hat = net(x1_train, x2_train)
            loss = loss_fn(y_hat, label)
            loss.backward()
            optimizer.step()

            res = [int(torch.equal(label[i], torch.round(y_hat)[i])) for i in range(0, len(x1_train))]
            acc = np.sum(res)/len(x1_train)
            if i % 10 == 0:
                sys.stdout.write("\r Train: Epoch number {}/10, Current loss {}, accuracy{}".format(epoch, loss.item(), acc))
                sys.stdout.flush()
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
        for i, data in enumerate(validation_loader, 0):
            net.eval()
            x1_val, x2_val, label = data
            x1_val, x2_val, label = x1_val.type(torch.FloatTensor), x2_val.type(torch.FloatTensor), label.type(torch.FloatTensor)
            x1_val, x2_val, label = x1_val.cuda(), x2_val.cuda(), label.cuda()
            y_hat = net(x1_val, x2_val)
            loss = loss_fn(y_hat, label)
            res = [int(torch.equal(label[i], torch.round(y_hat)[i])) for i in range(0, len(x1_val))]
            acc = np.sum(res)/len(x1_val)
            if i % 10 == 0:
                print("\r *********Test: Epoch number {}/10, Current loss {}, accuracy {}".format(epoch, loss.item(), acc))
                # sys.stdout.flush()
    show_plot(counter, loss_history)
    torch.save(net.state_dict(), 'gsp.pt')

    #     # print(loss)
    # torch.save(cnn3d, 'cnn3d.pt')

#
# def train(num_epochs):
#     dataset = CarpDataset(root='C:/Users/louxi/Desktop/icra2021/')
#     batch_size = 64
#     validation_split = .1
#     shuffle_dataset = True
#     # random_seed = 43
#
#     # Creating data indices for training and validation splits:
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(validation_split * dataset_size))
#     print('%d data for training, %d data for validation.' % (split, dataset_size-split))
#     if shuffle_dataset:
#         # np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_indices, val_indices = indices[split:], indices[:split]
#
#     # Creating data samplers and loaders:
#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)
#
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=train_sampler)
#     validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                     sampler=valid_sampler)
#     # Load model and train
#     net = CNN3d().cuda()
#     net.train()
#     print('Number of parameters: %d' % count_parameters(net))
#     loss_fn = torch.nn.BCELoss(reduction='mean')
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
#     counter = []
#     loss_history = []
#     iteration_number = 0
#     for epoch in range(num_epochs):
#         for i, data in enumerate(train_loader, 0):
#             net.train(True)
#             x_train, label = data
#             batch_size = x_train.size()
#             x_train, label = x_train.type(torch.FloatTensor), label.type(torch.FloatTensor)
#             x_train, label = x_train.reshape((batch_size[0], 1, 32, 32, 32)).cuda(), label.reshape((batch_size[0], 1)).cuda()
#             y_hat = net(x_train)
#             loss = loss_fn(y_hat, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             res = [int(torch.equal(label[i], torch.round(y_hat)[i])) for i in range(0, len(x_train))]
#             acc = np.sum(res)/len(x_train)
#             if i % 10 == 0:
#                 sys.stdout.write("\r Train: Epoch number {}/10, Current loss {}, accuracy{}".format(epoch, loss.item(), acc))
#                 sys.stdout.flush()
#                 iteration_number += 10
#                 counter.append(iteration_number)
#         validation_res = []
#         validation_loss = []
#         for i, data in enumerate(validation_loader, 0):
#             net.eval()
#             x_val, label = data
#             batch_size = x_val.size()
#             x_val , label = x_val.type(torch.FloatTensor), label.type(torch.FloatTensor)
#             x_val , label = x_val.reshape((batch_size[0], 1, 32, 32, 32)).cuda(), label.reshape((batch_size[0], 1)).cuda()
#             y_hat = net(x_val)
#             loss = loss_fn(y_hat, label)
#             validation_loss.append(loss.item())
#             batch_res = [int(torch.equal(label[i], torch.round(y_hat)[i])) for i in range(0, len(x_val))]
#             validation_res = validation_res + batch_res
#         print('Validation loss: {}, validation accuracy: {}'.format(np.average(validation_loss), np.sum(validation_res)/len(validation_res)))
#     torch.save(net.state_dict(), 'collision_1012.pt')
#     show_plot(counter, loss_history)
#
#
#     #     # print(loss)
#     # torch.save(cnn3d, 'cnn3d.pt')

if __name__ == '__main__':
    train_gsp(50)
