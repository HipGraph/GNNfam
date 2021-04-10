import argparse
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

import numpy as np

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.parse_args()


def full_pipeline(args):
    
    startpt = []
    endpt = []
    eweights = []

    with open(args.graph, "r") as f:
        ct = 0
        for line in f.readlines():
            a,b,c = line.strip().split()
            startpt.append(int(a))
            endpt.append(int(b))
            eweights.append(float(c))
            ct+=1
        print(ct)
    print(len(startpt), len(endpt))

    labels = []

    with open(args.labels, "r") as f:
        for l in f.readlines():
            labels.append(int(l.strip().split()[1]))
    print("Done reading")
    num_classes = len(set(labels))


    def convert_to_14_bit(num):
        #converts input from class to binary value
        #using this representation instead of one hot encoding as to save up on memory
        #may need to change this function if classes cannot be represented in 14 bits
        return list(map(int, list('{0:014b}'.format(num))))

    edges = torch.tensor([startpt, endpt], dtype=torch.long)
    y = torch.tensor(labels)
    if args.one_indexed_classes:
        y -= 1

    x = [convert_to_14_bit(val) for val in labels]
    data = Data(
        x=torch.Tensor(x), edge_index=edges, 
        y=y, num_classes=num_classes
    )
    print("Done Prepping data")

    nodesize = y.shape[0]
    trainsize = int(0.8*nodesize)
    testsize = nodesize - trainsize
    mask_split = []

    print("Creating train and test masks")

    with open(args.mask, 'rb') as f:
        mask_split = pickle.load(f)
        mask_split = np.asarray(mask_split)
        data.train_mask = mask_split == 1
        data.test_mask = mask_split == 2

    data.train_mask = torch.from_numpy(data.train_mask)
    data.test_mask = torch.from_numpy(data.test_mask)
    print(data.train_mask)
    print(data.test_mask)

    class Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, concat=False):
            super(Net, self).__init__()
            self.conv1 = SAGEConv(in_channels, 16)
            self.conv2 = SAGEConv(16, out_channels)

        def forward(self, data):
            x = F.relu(
                self.conv1(data.x, data.edge_index)
            )
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, data.edge_index)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_bits = 14
    model = Net(in_bits,num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    data = data.to(device)

    def train():
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return total_loss


    def test():
        model.eval()
        correct = 0
        _, pred = model(data).max(dim=1)
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / data.test_mask.sum().item()

    for epoch in range(1, 401):
        
        loss = train()
        test_acc = test()
        print(
            'Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
                epoch, loss, test_acc
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph", required=True, help="path to graph file", type=str)
    parser.add_argument(
        "--labels", required=True, help="path to labels file", type=str
    )
    parser.add_argument(
        "--mask", required=True, help="path to mask pickle file", type=str
    )
    parser.add_argument(
        "--one_indexed_classes", required=False,
        help="to use when labels are 1 indexed instead of 0", 
        dest='one_indexed_classes', action='store_true'
    )
    parser.set_defaults(one_indexed_classes=False)

    args = parser.parse_args()
    full_pipeline(args)
