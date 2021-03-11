import pickle

import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

import numpy as np

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


def convert_to_14_bit(num):
    return list(map(int, list('{0:014b}'.format(num))))


startpt = []
endpt = []
eweights = []
with open("Graph.txt", "r") as f:
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

with open("label.txt", "r") as f:
    for l in f.readlines():
        labels.append(int(l.strip().split()[1]))
print("Done reading")
num_classes = len(set(labels))

for i in range(-5, 6, 1):
    seed_val = 42 + i
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    edges = torch.tensor([startpt, endpt], dtype=torch.long)
    y = torch.tensor(labels)
    y -= 1
    x = [convert_to_14_bit(val) for val in labels]
    data = Data(x=torch.Tensor(x), edge_index=edges, y=y)
    print("Done Prepping data")

    nodesize = y.shape[0]
    trainsize = int(0.8*nodesize)
    testsize = nodesize - trainsize

    edges = torch.tensor([startpt, endpt], dtype=torch.long)
    y = torch.tensor(labels)
    y -= 1 
    x = [convert_to_14_bit(val) for val in labels]
    data = Data(x=torch.Tensor(x), edge_index=edges, y=y)
    print("Done Prepping data")

    nodesize = y.shape[0]
    trainsize = int(0.8*nodesize)
    testsize = nodesize - trainsize
    mask_split = []

    print("Creating train and test masks")

    with open('mask_split.pkl', 'rb') as f:
        mask_split = pickle.load(f)
        mask_split = np.asarray(mask_split)
        data.train_mask = mask_split == 1
        data.test_mask = mask_split == 2

    data.train_mask = torch.from_numpy(data.train_mask)
    data.test_mask = torch.from_numpy(data.test_mask)

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

    print("Training network with seed {}".format(seed_val))
    total_epochs = 400
    for epoch in range(1, total_epochs+1):
        loss = train()
    print("Trained network for {} epochs".format(total_epochs))
    data.test_mask[data.test_mask != True] =  True
    data = data.to(device)
    test_acc, pred = test()
    print('Seed Value: {} Test Acc: {:.4f}'.format(seed_val, test_acc))
    print("----------------------------------------------")