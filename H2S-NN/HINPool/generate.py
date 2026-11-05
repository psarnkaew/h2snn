#################################################################
# 
# This file is used to generate bin of train/val/test split
# 
#################################################################
import argparse
import numpy as np
import os, sys
import torch
import dgl
import random
import networkx as nx


from utils.utils import *
import torch.utils.data
from dgl.data import TUDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from HeteroDataset import HeteroDataset

def arg_parse():
    """
    argument parser
    """
    parser = argparse.ArgumentParser(description="Data generation arguments")
    parser.add_argument(
        "--dataset", 
        dest="dataset", 
        help="Input Dataset"
    )
    parser.add_argument(
        "--num-ntype", 
        dest="num_ntype", 
        type=int,
        help="Number of atom types to be transformed into node types"
    )
    parser.add_argument(
        "--hetero-mode", 
        dest="hetero_mode", 
        type=int,
        help="how heterogeneous will this dataset be"
    )
    parser.add_argument(
        "--seed", 
        dest="seed", 
        type=int, 
        help="training seed"
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float
    )
    parser.add_argument(
        "--test-ratio",
        dest="test_ratio",
        type=float
    )
    parser.set_defaults(
        dataset="MUTAG",
        num_ntype=0,
        hetero_mode=0,
        train_ratio=0.7,
        test_ratio=0.2
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Specify setting
    prog_args = arg_parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TUDataset(name=prog_args.dataset)

    # Get the maximum node & edge label (Used for one-hot encoding) 
    max_node_label, max_edge_label = 0, 0
    for i in range(len(dataset)):
        # Node types part
        graph, label = dataset[i]
        local_max_node = int(graph.ndata['node_labels'].max())
        max_node_label = local_max_node if local_max_node > max_node_label \
                                        else max_node_label
        # Edge types part
        if 'edge_labels' not in graph.edata.keys():
            dataset[i][0].edata['edge_labels'] = torch.zeros(graph.number_of_edges(), dtype=torch.long)
            pass
        else:
            local_max_edge = int(graph.edata['edge_labels'].max())
            max_edge_label = local_max_edge if local_max_edge > max_edge_label \
                                        else max_edge_label
            
    # Transform node label information into one-hot encoding features
    for idx, graph in enumerate(dataset):
        node_labels = graph[0].ndata['node_labels']
        if 'node_attr' in graph[0].ndata:
            graph[0].ndata['feat'] = graph[0].ndata.pop('node_attr').to(torch.float32)
        ########################
        if 'feat' not in graph[0].ndata:
            graph[0].ndata['feat'] = torch.zeros((len(node_labels), max_node_label + 1))
            graph[0].ndata['feat'][torch.arange(len(node_labels)), node_labels[:, 0]] = 1
        ########################
    
    # (Only for NCIs) Counting number of atoms and get node label mask
    mask = None
    if prog_args.num_ntype > 0:
        hist = np.zeros((max_node_label+1,))
        for i in range(len(dataset)):
            g, _ = dataset[i]
            n_label = g.ndata['node_labels'].flatten()
            # print(n_label)
            unique_elements, counts = n_label.unique(return_counts=True)
            # print(unique_elements, counts)
            for idx, ele in enumerate(unique_elements):
                hist[ele] += counts[idx]
        topk_atoms = np.argsort(hist)[::-1][:prog_args.num_ntype]
        topk_atoms = np.ascontiguousarray(topk_atoms)
        print("TopK atoms", topk_atoms)
        # Create a mask for nodes with labels 0, 2, 4, 6, 8
        mask = torch.zeros(max_node_label + 1, dtype=torch.int64) + prog_args.num_ntype
        mask[topk_atoms] = torch.arange(prog_args.num_ntype)
    print("Mask:", mask)
    print(f'Max node label: {max_node_label}\nMax edge label: {max_edge_label}')
    
    # Start constructing HINs (-> HeteroDataset.py)
    print("Start Constructing")
    hetero_dataset = HeteroDataset(
        dataset=dataset, 
        name=prog_args.dataset, 
        mode=prog_args.hetero_mode,
        max_node_label=max_node_label,
        max_edge_label=max_edge_label,
        mask=mask,
        device=device,
    )
    print("End Constructing")
    
    # Save into files to be used later
    os.makedirs(f'dataset/{prog_args.dataset}', exist_ok=True)
    start_save = time.time()
    if prog_args.num_ntype > 0:
        save_graphs(f'dataset/{prog_args.dataset}/{prog_args.num_ntype}.pkl', hetero_dataset.graphs, {'glabel': hetero_dataset.labels})
    else:
        save_graphs(f'dataset/{prog_args.dataset}/all_type.pkl', hetero_dataset.graphs, {'glabel': hetero_dataset.labels})
    
    print(f'Save graph using {time.time() - start_save}')
 