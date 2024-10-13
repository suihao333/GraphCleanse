from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from aug import TUDataset as TUDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle
import random


class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def train(epoch, use_unsup_loss):
    loss_fn = F.cross_entropy
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    if use_unsup_loss:
        for data, data2 in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()

            sup_loss = loss_fn(model(data), data.y, reduction='sum')
            unsup_loss = model.unsup_loss(data2)
            if separate_encoder:
                unsup_sup_loss = model.unsup_sup_loss(data2)
                loss = sup_loss + unsup_loss + unsup_sup_loss * lamda
            else:
                loss = sup_loss + unsup_loss * lamda

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            if separate_encoder:
                unsup_sup_loss_all += unsup_sup_loss.item()
            loss_all += loss.item() * data.num_graphs

            optimizer.step()

        if separate_encoder:
            print("sup_loss_all, unsup_loss_all, unsup_sup_loss_all",sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
        else:
            print("sup_loss_all, unsup_loss_all",sup_loss_all, unsup_loss_all)
        return loss_all / len(train_loader.dataset)
    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            sup_loss = loss_fn(model(data), data.y, reduction='sum')
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()


def test(loader):
    model.eval()
    error = 0
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred = predict_fn(pred)
        y_true.append(data.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from model import Net
    from arguments import arg_parse 
    args = arg_parse()

    # ============
    # Hyperparameters
    # ============
    target = args.target
    dim = 64
    epochs = 100
    batch_size = 20
    lamda = args.lamda
    use_unsup_loss = args.use_unsup_loss
    separate_encoder = args.separate_encoder

    dataset= TUDataset('../dataset/AIDS', 'AIDS')
    print('num_features : {}\n'.format(dataset.num_features))


    graph_ids = [i for i in range(len(dataset))]
    with open('/data/AIDS/AIDS/bkd/test_bkd_gids_label.pkl', 'rb') as file:
        test_bkd_gids_label = pickle.load(file)
    bkd_gids = [item[0] for item in test_bkd_gids_label]

    with open('/data/AIDS/AIDS/compared_same_samples.pkl', 'rb') as file:
        compared_same_samples = pickle.load(file)
    with open('/data/AIDS/AIDS/same_train_bkd.pkl', 'rb') as file:
        same_train_bkd = pickle.load(file)
    bkd_gids = sorted(bkd_gids)
    bkd_gids_subset_tensor = torch.tensor(bkd_gids)
    test_ids = bkd_gids

    train_valid_list = [item for item in graph_ids if item not in test_ids]
    sup_train_list = [item for item in compared_same_samples if item not in test_ids]
    all_train_ids = random.sample(train_valid_list, 1600)
    valid_ids = list(set(train_valid_list) - set(all_train_ids))
    sup_train_id = sup_train_list
    num_graphs = len(dataset)

    test_dataset = dataset[test_ids]
    val_dataset = dataset[valid_ids]
    train_dataset = dataset[sup_train_id]


    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    if use_unsup_loss:
        unsup_train_dataset = dataset[all_train_ids]
        unsup_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True)

        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

    val_error = test(val_loader)
    test_error = test(test_loader)
    print('Epoch: {:03d}, Validation ACC: {:.7f}, Test ACC: {:.7f},'.format(0, val_error, test_error))

    best_val_error = None
    for epoch in range(1, epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch, use_unsup_loss)
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error


        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation ACC: {:.7f}, '
              'Test ACC: {:.7f},'.format(epoch, lr, loss, val_error, test_error))

    with open('supervised.log', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(target,args.train_num,use_unsup_loss,separate_encoder,args.lamda,args.weight_decay,val_error,test_error))
