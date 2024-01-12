import pickle
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import torch
from sklearn import metrics
from data_loader import load_net_specific_data
import torch.nn.functional as F
parser = argparse.ArgumentParser()
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
    return data
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='fe_CDG/ECD-CDGI/data/GGNet/dataset_GGNet_ten_5CV.pkl',
                    help='The path of the input pkl file.')  # When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()
device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
data = load_net_specific_data(args)
print(data)