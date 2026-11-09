import torch
from torch_geometric.datasets import HGBDataset
from torch_geometric.utils import to_undirected
d = HGBDataset(root=r"C:\Users\Acer\Desktop\h2snn\H2S-NN\data\freebase", name="Freebase")[0]
y = d["book"].y
ei = to_undirected(d["book", "and", "book"].edge_index, num_nodes=d["book"].num_nodes)
r, c = ei
lab = (y[r] >= 0) & (y[c] >= 0)
loop = r == c
for name, m in (("labelled, loops kept", lab), ("labelled, loops dropped", lab & ~loop)):
    print(name, int(m.sum()), "H_edge(heterophily) =", round(1 - float((y[r][m] == y[c][m]).double().mean()), 4))
print("labelled loops:", int((lab & loop).sum()))
