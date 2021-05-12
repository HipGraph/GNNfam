import pickle

import networkx as nx
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, NeighborSampler

import numpy as np

np.random.seed(42)
torch.manual_seed(42)

startpt = []
endpt = []
eweights = []
with open("deepfamgraph.txt", "r") as f:
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

with open("deepfamlabel.txt", "r") as f:
    for l in f.readlines():
        labels.append(int(l.strip().split()[1]))
print("Done reading")
num_classes = len(set(labels))

def convert_to_14_bit(num):
    return list(map(int, list('{0:014b}'.format(num))))

edges = torch.tensor([startpt, endpt], dtype=torch.long)
y = torch.tensor(labels)
# y -= 1 
x = [convert_to_14_bit(val) for val in labels]
data = Data(x=torch.Tensor(x), edge_index=edges, y=y, num_classes=num_classes)
print("Done Prepping data")

nodesize = y.shape[0]
trainsize = int(0.8*nodesize)
testsize = nodesize - trainsize
mask_split = []

print("Creating train and test masks")

with open('deepfammask_split.pkl', 'rb') as f:
    mask_split = pickle.load(f)
    mask_split = np.asarray(mask_split)
    data.train_mask = mask_split == 1
    data.test_mask = mask_split == 2

data.train_mask = torch.from_numpy(data.train_mask)
data.test_mask = torch.from_numpy(data.test_mask)
print(data.train_mask)
print(data.test_mask)

print("Creating NeighborSampler for training nodes")
sample_loader = NeighborSampler(
    data.edge_index, node_idx=data.train_mask, sizes=[25, 10], 
    num_nodes=len(data.y), batch_size=512, shuffle=True, num_workers=12
)

print("Creating NeighborSampler all nodes")
subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None, sizes=[-1],
    num_nodes=len(data.y), batch_size=512, shuffle=False, num_workers=12
)

with open('deepfam_pyg_data_obj.pkl', 'wb') as f:
    torch.save(data, f)

with open('deepfam_pyg_train_sample_loader.pkl', 'wb') as f:
    torch.save(sample_loader, f)

with open('deepfam_pyg_complete_sample_loader.pkl', 'wb') as f:
    torch.save(subgraph_loader, f)

data = None
with open('deepfam_pyg_data_obj.pkl', 'rb') as f:
    data = torch.load(f)

sample_loader = None 
with open('deepfam_pyg_train_sample_loader.pkl', 'rb') as f:
    sample_loader = torch.load(f)

subgraph_loader = None
with open('deepfam_pyg_complete_sample_loader.pkl', 'rb') as f:
    subgraph_loader = torch.load(f)

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(Net, self).__init__()
        
        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, 16))
        self.convs.append(SAGEConv(16, out_channels))


    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        return x_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_bits = 14
model = Net(in_bits,data.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()

    total_loss = total_correct = 0
    for ix, sample in enumerate(sample_loader):
        batch_size, n_id, adjs = sample
        # print("batch {} of {}".format(ix, len(sample_loader)))
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(
            data.y[n_id[:batch_size]].to(device)
        ).sum())

    loss = total_loss / len(sample_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc



@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)

    y_true = data.y.unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results

for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')