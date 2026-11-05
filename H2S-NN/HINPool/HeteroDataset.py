import os
import gc
import torch
import dgl
import numpy as np
import time
from torch.utils.data import Dataset
from utils.utils import *
from tqdm import tqdm

class HeteroDataset(Dataset):
    def __init__(self, dataset, name, mode, max_node_label, max_edge_label, mask, device):
        self.data = dataset
        self.name = name
        self.mode = mode
        if mask is None:
            num_ntype = max_node_label
            num_etype = max_edge_label
        else:
            num_ntype = np.unique(mask).max()
            num_etype = max_edge_label
        # Assign node & edge type name
        # 0 for hetero nodes & hetero edges
        if self.mode == 0:
            self.ntype = np.array([f'n_{i}' for i in range(num_ntype + 1)])
            self.etype = np.array([f'e_{i}' for i in range(num_etype + 1)])
        # 1 for hetero nodes & homo edges
        elif self.mode == 1:
            self.ntype = np.array([f'n_{i}' for i in range(num_ntype + 1)])
            self.etype = np.array(['-'])
        # 2 for homo nodes & hetero edges
        elif self.mode == 2:
            self.ntype = np.array(['n'])
            self.etype = np.array([f'e_{i}' for i in range(num_etype + 1)])
        # 3 for homo nodes & homo edges
        elif self.mode == 3:
            self.ntype = np.array(['n'])
            self.etype = np.array(['-'])
            
        # Permute node & edge type to obtain a whole view within HIN
        self.metagraph = permutation(self.ntype, self.etype)
        # Store node type and edge type information
        self.node_dict, self.edge_dict = get_dict(self.ntype, self.etype, self.metagraph)
        
        self.graphs = self.data[:][0]  # Graphs
        self.labels = self.data[:][1]  # Graph clasiification labels (Not node or edge labels)
        for graph_idx in tqdm(range(len(self.graphs))):
            ############ Check zero-degree and Drop nodes with zero degree ################
            in_degrees = self.graphs[graph_idx].in_degrees()
            out_degrees = self.graphs[graph_idx].out_degrees()

            nodes_with_zero_degree = (in_degrees == 0) & (out_degrees == 0)
            nodes_with_zero_degree_indices = nodes_with_zero_degree.nonzero().squeeze(1)

            self.graphs[graph_idx] = dgl.remove_nodes(self.graphs[graph_idx], nodes_with_zero_degree_indices)
            ###############################################################################
            # Modify node labels with less count to 'others'
            if mask is not None:
                node_labels = self.graphs[graph_idx].ndata['node_labels']
                node_labels = mask[node_labels]
                self.graphs[graph_idx].ndata['node_labels'] = node_labels
        
        # Batch graphs for construction efficiency (-> utils.transform batch)
        all_graph = dgl.batch(self.graphs)
        start = time.time()
        self.graphs = transform_batch(
                        all_graph, 
                        self.name, 
                        self.mode, 
                        self.ntype, 
                        self.etype,
                        self.node_dict, 
                        self.edge_dict, 
                        self.metagraph)
        print(f'Transform using {time.time() - start :.2f} secs')
        print(len(self.graphs), 'heterogeneous graphs in total')

    def __getitem__(self, index):
        # Get constructed HINs and their graph clasiification labels
        graph, label = self.graphs[index], self.labels[index]

        return (graph, label)
    
    def __len__(self):
        return len(self.data)

class myDataset(Dataset):
    def __init__(self, graphs, labels, args):
        if args.backbone == 'HGT':
            self.graphs = graphs
        if args.backbone == 'Simple':
            self.graphs = graphs
            print(f"Preprocessing: Convertion & gid")
            if args.dataset in ['NCI1', 'NCI109']:
                for i in tqdm(range(len(graphs))):
                    self.graphs[i] = dgl.to_homogeneous(self.graphs[i], ndata=['feat'], edata=['id'])
                    self.graphs[i].ndata['gid'] = torch.ones((self.graphs[i].num_nodes(),), dtype=torch.long, device=self.graphs[0].device) * i
                    # self.graphs[i].edata['id'] = torch.zeros_like(self.graphs[i].edata['id'])
                print(graphs[0].edata)
            else:
                for i in tqdm(range(len(graphs))):
                    self.graphs[i] = dgl.to_homogeneous(self.graphs[i], ndata=['feat'], edata=['id'])
                    self.graphs[i].ndata['gid'] = torch.ones((self.graphs[i].num_nodes(),), dtype=torch.long, device=self.graphs[0].device) * i
        self.labels = labels
        self.args   = args

    def __getitem__(self, index):
        graph, label = self.graphs[index], self.labels[index]

        return (graph, label)
    
    def __len__(self):
        return len(self.graphs)