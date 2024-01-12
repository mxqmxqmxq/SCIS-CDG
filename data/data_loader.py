import pickle
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
import torch
import numpy as np
from torch_geometric.utils.loop import add_self_loops
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
def load_obj( name ):
    """
    Load dataset from pickle file.
    :param name: Full pathname of the pickle file
    :return: Dataset type of dictionary
    """
    with open( name , 'rb') as f:
        return pickle.load(f)


# When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
# args.dataset_file='./data/GGNet/dataset_GGNet_ten_5CV.pkl'
# args.is_5_CV_test = Ture

def load_net_specific_data(args):
    """
    Load network-specific dataset from the pickle file.
    :param args: Arguments received from command line
    :return: Data for training model (class: 'torch_geometric.data.Data')
    """
    dataset = load_obj(args.dataset_file)

    std = StandardScaler()
    features = std.fit_transform(dataset['feature'].detach().numpy())
    features = torch.FloatTensor(features)
    
    if args.is_5_CV_test:
        mask = dataset['split_set']
    else:
        mask = dataset['mask']
    data = Data(x=features, y=dataset['label'], edge_index=dataset['edge_index'], mask=mask, node_names=dataset['node_name'])
     # get compressed adj
    edges, weight = get_laplacian(edge_index=data.edge_index, normalization='sym', num_nodes=data.x.shape[0])
    edges, weight = add_self_loops(edge_index=edges, fill_value=1., edge_attr=-weight, num_nodes=data.x.shape[0])
    adj = torch.sparse_coo_tensor(indices=data.edge_index, 
                                    values=torch.ones_like(data.edge_index[0]),
                                    size=(data.x.shape[0], data.x.shape[0]), 
                                    device=data.x.device,
                                    dtype=torch.float).to_dense()
    adj = F.normalize(adj, dim=1)
    U, S, V = torch.svd_lowrank(adj, q=args.nl)
    adj = torch.mm(U, torch.diag(S))
    
    adj = F.normalize(adj, dim=0)
    data.adj = adj
    return data











