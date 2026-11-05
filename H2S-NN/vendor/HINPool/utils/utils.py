import gc
import math
import random
import time
from itertools import permutations, product

import dgl
import dgl.backend as F
import networkx as nx
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def stratified_split(dataset, train_size, val_size, test_size, random_state):
    from sklearn.model_selection import (
        StratifiedShuffleSplit, train_test_split
    )

    print(train_size, val_size, test_size)

    features = np.array(dataset.graphs)
    labels = np.array(dataset.labels)

    num_splits = 1

    # STRATIFIED TESTING SET
    stratified_splitter = StratifiedShuffleSplit(n_splits=num_splits, \
        test_size=test_size, random_state=random_state)

    # Perform the split
    for train_val_index, test_index in stratified_splitter.split(
        features, labels
    ):
        train_val_features, test_features = features[train_val_index
                                                     ], features[test_index]
        train_val_labels, test_labels = labels[train_val_index], labels[
            test_index]

        train_val_set = torch.utils.data.Subset(dataset, train_val_index)
        test_set = torch.utils.data.Subset(dataset, test_index)


    stratified_splitter_validation = StratifiedShuffleSplit(n_splits=num_splits, \
        test_size=val_size, random_state=random_state)

    for train_index, val_index in stratified_splitter_validation.split(
        train_val_features, train_val_labels
    ):
        train_set = torch.utils.data.Subset(train_val_set, train_index)
        val_set = torch.utils.data.Subset(train_val_set, val_index)

    # These are to be saved once
    orig_train_index = train_val_index[train_index]
    orig_val_index = train_val_index[val_index]
    orig_test_index = test_index

    return train_set, val_set, test_set, orig_train_index, orig_val_index, orig_test_index


def plot_loss(train_losses, val_losses, args, i, full_path):
    x = np.arange(len(train_losses))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, train_losses, color='blue', linewidth=2, label='train')
    plt.plot(x, val_losses, color='red', linewidth=2, label='validation')
    plt.legend()
    plt.title('Loss')
    path_name = (
        '/loss_'
        f'{args.lr}_'
        f'{args.num_heads}_'
        f'{args.hidden_dim}_'
        f'{args.batch_size}_{i}'
        '.png'
    )
    plt.savefig(full_path + path_name)


def plot_acc(train_acc, val_acc, test_acc, args, i, full_path):
    x = np.arange(len(train_acc))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.yticks(np.arange(30, 101, 5))
    plt.plot(x, train_acc, color='blue', label='train')
    plt.plot(x, val_acc, color='red', label='validation')
    plt.plot(x, test_acc, color='green', label='test')
    plt.legend()
    plt.title('Accuracy')
    path_name = (
        '/acc_'
        f'{args.lr}_'
        f'{args.num_heads}_'
        f'{args.hidden_dim}_'
        f'{args.batch_size}_{i}'
        '.png'
    )
    plt.savefig(full_path + path_name)


def plot_lr(lrs, full_path):
    x = np.arange(len(lrs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, lrs, color='red', label='train')
    plt.legend()
    plt.title('Learning rate')
    plt.savefig(full_path + '/lr.png')


def same_seeds(seed):  # Fix random seeds for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def permutation(ntypes, etypes):
    # Returns a networkX graph as metagraph
    edge_list = []
    for i in list(product(ntypes, ntypes)):
        for j in etypes:
            etype_name = "{}-{}-{}".format(i[0], j, i[1])
            edge_list.append((*i, etype_name))
            # edge_list.append(i)
    G = nx.MultiDiGraph()
    G.add_nodes_from(ntypes)
    G.add_edges_from(edge_list)
    # print("permutation", edge_list)
    return G


def get_dict(ntypes, etypes, metagraph):
    node_dict, edge_dict = {}, {}
    for idx, ntype in enumerate(ntypes):
        node_dict[ntype] = idx
    count = 0
    for idx, metagraph_etype in enumerate(metagraph.edges()):
        # print("get_dict", "{}-{}-{}".format(metagraph_etype[0], etypes[count], metagraph_etype[1]))
        etype_name = "{}-{}-{}".format(
            metagraph_etype[0], etypes[count], metagraph_etype[1]
        )
        edge_dict[etype_name] = idx
        if count == len(etypes) - 1:
            count = 0
        else:
            count += 1

    return node_dict, edge_dict


def check_zero_degree(g, layer):
    g = dgl.to_homogeneous(g)
    ######################### Check zero-degree #########################
    in_degrees = g.in_degrees()
    out_degrees = g.out_degrees()

    nodes_with_zero_degree = (in_degrees == 0) & (out_degrees == 0)
    nodes_with_zero_degree_indices = nodes_with_zero_degree.nonzero(
    ).squeeze(1)

    print(
        f'Layer {layer}, Nodes with zero degree: {len(nodes_with_zero_degree_indices)}/{g.num_nodes()}'
    )


def transform_batch(
    g, name, mode, ntypes, etypes, node_dict, edge_dict, metagraph
):
    # Preprocess
    g.ndata['node_labels'] = torch.reshape(g.ndata['node_labels'], (-1, ))

    # Get the source and destination nodes for all edges
    src, dst = g.edges()

    # Determine node types based on heterogeneous mode in HeteroDataset
    # If mode is 0 or 1, use heterogeneous node types
    if mode == 0 or mode == 1:
        src_nodes = ntypes[g.ndata['node_labels'][src]]
        dst_nodes = ntypes[g.ndata['node_labels'][dst]]
    # For others, use homogeneous node types ('n' for all nodes)
    else:
        src_nodes = ['n' for _ in src]
        dst_nodes = ['n' for _ in dst]
        g.ndata['node_labels'] = torch.zeros(
            g.number_of_nodes(), dtype=torch.long
        )

    edge_labels = torch.Tensor([])
    zipped_edges = list(zip(src_nodes, dst_nodes))

    # Determine edge types based on heterogeneous mode in HeteroDataset
    # If mode is 0 or 2, use heterogeneous edge types
    if mode == 0 or mode == 2:
        for idx, nodes in tqdm(enumerate(zipped_edges)):
            # Construct heterogeneous edge type name (source_type - edge_type - destination_type)
            etype_name = "{}-{}-{}".format(
                nodes[0], etypes[g.edata['edge_labels'][idx].item()], nodes[1]
            )
            edge_labels = torch.cat(
                (edge_labels, torch.Tensor([edge_dict[etype_name]])), 0
            )  # All set to 0
    # For others, use homogeneous edge types ('e' for all edges)
    else:
        for idx, nodes in enumerate(zipped_edges):
            etype_name = "{}---{}".format(nodes[0], nodes[1])
            edge_labels = torch.cat(
                (edge_labels, torch.Tensor([edge_dict[etype_name]])), 0
            )

    # Store the new edge labels in the graph
    g.edata['new_edge_labels'] = edge_labels

    # Convert the graph to a heterogeneous graph using the given node types and edge types
    hg = dgl.to_heterogeneous(
        g, ntypes=ntypes, etypes=list(edge_dict.keys()),
        ntype_field='node_labels', etype_field='new_edge_labels',
        metagraph=metagraph
    )

    # Prepare node and edge labels for each graph in the batch
    num_nodes = g.batch_num_nodes().numpy()
    num_nodes = np.insert(np.cumsum(num_nodes), 0, 0)
    nlabel_each_graph = [
        g.ndata['node_labels'][num_nodes[i]:num_nodes[i + 1]]
        for i in range(len(num_nodes) - 1)
    ]

    num_edges = g.batch_num_edges().numpy()
    num_edges = np.insert(np.cumsum(num_edges), 0, 0)
    elabel_each_graph = [
        edge_labels[num_edges[i]:num_edges[i + 1]]
        for i in range(len(num_edges) - 1)
    ]

    # Swap keys and values in node_dict and edge_dict for easy lookup under
    node_dict_swap = {v: k for k, v in node_dict.items()}
    edge_dict_swap = {v: k for k, v in edge_dict.items()}

    # Initialize tensors to store the number of each type of node and edge per graph in the batch
    node_batch = {
        ntype: torch.zeros(len(nlabel_each_graph)).int()
        for ntype in node_dict
    }
    edge_batch = { (etype.split('-')[0], etype, etype.split('-')[-1])\
        :torch.zeros(len(elabel_each_graph)).int() for etype in edge_dict}

    for idx in range(len(nlabel_each_graph)):
        # Count the occurrences of each node and edge type in each graph
        node_key, node_count = torch.unique(
            nlabel_each_graph[idx], return_counts=True
        )
        edge_key, edge_count = torch.unique(
            elabel_each_graph[idx], return_counts=True
        )
        node_key, edge_key = node_key.int().tolist(), edge_key.int().tolist()

        # Update the node and edge batch counts
        for key, count in zip(node_key, node_count):
            node_batch[node_dict_swap[key]][idx] += count
        for key, count in zip(edge_key, edge_count):
            etype = edge_dict_swap[key]
            edge_batch[etype.split('-')[0], etype,
                       etype.split('-')[-1]][idx] += count

    # Store dictionaries and edge IDs in the heterogeneous graph
    hg.node_dict = node_dict
    hg.edge_dict = edge_dict
    hg.edata['id'] = hg.edata.pop('_ID')
    for etype in hg.edge_dict:
        hg.edges[etype].data['id'] = \
                    torch.ones(hg.number_of_edges(etype), dtype=torch.long)\
                    * hg.edge_dict[etype]

    # Set the number of nodes and edges for each type per graph in the batch for unbatching
    hg.set_batch_num_nodes(node_batch)
    hg.set_batch_num_edges(edge_batch)

    hgs = dgl.unbatch(hg)
    for g_idx in range(len(hgs)):
        hgs[g_idx].node_dict = node_dict
        hgs[g_idx].edge_dict = edge_dict

    return hgs


def my_load_graph(dataset, num_ntype):
    start_load = time.time()
    if num_ntype > 0:
        print(f'Loading graphs from dataset/{dataset}/{num_ntype}.pkl')
        graphs, labels = load_graphs(f'dataset/{dataset}/{num_ntype}.pkl')
    else:
        print(f'Loading graphs from dataset/{dataset}/all_type.pkl')
        graphs, labels = load_graphs(f'dataset/{dataset}/all_type.pkl')
    print(f'Load graph using {time.time() - start_load :.2f} sec')
    input_dim = graphs[0].ndata['feat']['n_0'].shape[1]
    label_dim = 2 if dataset != 'ENZYMES' else 6
    return graphs, input_dim, labels['glabel'], label_dim


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
    num_cycles: float = 0.5, last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


from functools import reduce
from typing import List


def add_nhop_edges_beforehand(g, n: List[int]):
    """Add nhop edges to the given graph. The n-hop edges contain no self-loop,
        and no added edges connect to the same node pair.

    Args:
        g (DGLGraph | DGLHeteroGraph): Either heterogeneous and homogeneous. If it is heterogenous,
            it would be first transformed to homogeneous via `dgl.to_homogeneous`.
        n (int): n hops. n >= 2

    Returns:
        A homogeneous graph with the added edges have a new edge type, which can be access via
        `g.edata[dgl.ETYPE]`.
        Similarly, the eids are available through `g.edata[dgl.EID]`
    """
    n = n[-1]
    if g.is_homogeneous:
        g = g.clone()
    else:
        g = dgl.to_homogeneous(g)

    adj = g.adjacency_matrix().coalesce()
    old_indices = adj.indices()

    nhop_adj = reduce(lambda a, b: a @ b, [adj] * n)

    indices = nhop_adj.indices()
    indices = indices[:, indices[0] != indices[1]]  # exclude self-loops

    # Only add non-existing edge
    new_and_old_indices = torch.cat(
        [old_indices, indices], dim=1
    )  # Combine 1-hop and 2-hop indices.
    unique_rows, counts = \
        torch.unique(new_and_old_indices, dim=1, return_counts=True)  # Get unique indices and their counts.
    non_duplicate = (
        counts == 1
    )  # If the count is greater than 1, means it is an overlapping edge.
    indices = unique_rows[:, non_duplicate
                          ]  # So drop them and only get non-overlapping ones.

    # Edge data alignment
    edata = {}
    if dgl.ETYPE in g.edata:
        edata[dgl.ETYPE] = torch.full(
            (indices.shape[-1], ), fill_value=g.edata[dgl.ETYPE].max() + 1,
            device=g.device
        )
    if dgl.EID in g.edata:
        edata[dgl.EID] = torch.arange(indices.shape[-1], device=g.device)

    # Add edges (u, v, edata)
    g.add_edges(indices[0], indices[1], edata)

    return g


def add_nhop_edges_in_layer(g, n):
    """Add nhop edges to the given graph. The n-hop edges contain no self-loop,
        and no added edges connect to the same node pair.

    Args:
        g (DGLGraph | DGLHeteroGraph): Either heterogeneous and homogeneous. If it is heterogenous,
            it would be first transformed to homogeneous via `dgl.to_homogeneous`.
        n (int): n hops. n >= 2

    Returns:
        A homogeneous graph with the added edges have a new edge type, which can be access via
        `g.edata[dgl.ETYPE]`.
        Similarly, the eids are available through `g.edata[dgl.EID]`
    """
    n = n[0]
    if n == 1:
        return g
    assert n > 1
    if g.is_homogeneous:
        g = g.clone()
    else:
        g = dgl.to_homogeneous(g)

    batch_size = g.batch_size
    adj = g.adjacency_matrix().coalesce()
    old_indices = adj.indices()

    nhop_adj = reduce(lambda a, b: a @ b, [adj] * n)

    indices = nhop_adj.indices()
    indices = indices[:, indices[0] != indices[1]]  # exclude self-loops

    # Only add non-existing edge
    new_and_old_indices = torch.cat(
        [old_indices, indices], dim=1
    )  # Combine 1-hop and 2-hop indices.
    src_nodes = new_and_old_indices[
        0]  # Obtaining batch id only needs one side of nodes.
    src_nodes_batch = g.ndata['_BATCH'][
        src_nodes
    ]  # Change the indices into batch id, needs '_BATCH' in ndata.
    """ An example of injecting batch id
    batch_id = torch.tensor([], dtype=torch.long, device=g.device)
    for ntype in hg.ntypes:
        temp = hg.batch_num_nodes(ntype)
        for id in range(len(g.batch_num_nodes())):
            batch_id = torch.cat([batch_id, torch.full((temp[id],), id, device=g.device)])
    g.ndata['_BATCH'] = batch_id
    """

    # Filter out overlapping edges
    add_indices_all = [
    ]  # Obtain unique indices graph by graph to prevent meaningless comparison.
    for gid in range(batch_size):
        temp = new_and_old_indices[:, src_nodes_batch == gid
                                   ]  # Filter out edges in the i-th graph.
        unique_rows, counts = \
            torch.unique(temp, dim=1, return_counts=True)  # Get unique indices and their counts.
        non_duplicate = (
            counts == 1
        )  # If the count is greater than 1, means it is an overlapping edge.
        add_indices = unique_rows[:, non_duplicate
                                  ]  # So drop them and only get non-overlapping ones.
        add_indices_all.append(add_indices)
    indices = torch.cat(
        add_indices_all, dim=1
    )  # In this way those indices are non-overlapping 2-hop edges to be added.

    # Edge data alignment
    edata = {}
    if dgl.ETYPE in g.edata:
        edata[dgl.ETYPE] = torch.full(
            (indices.shape[-1], ), fill_value=g.edata[dgl.ETYPE].max() + 1,
            device=g.device
        )
    if dgl.EID in g.edata:
        edata[dgl.EID] = torch.arange(indices.shape[-1], device=g.device)
    """ 
    Since DGLGraph.add_edges(u,v, data) will discard batch information, 
        batch information needs to be updated and re-assign.
    Additionally, no nodes are modified, so just copy the batch_num_nodes()

    See note 5 in https://docs.dgl.ai/en/1.1.x/generated/dgl.DGLGraph.add_edges.html
    """
    # Batch information retrieval
    batch_num_nodes = g.batch_num_nodes()
    batch_num_edges = g.batch_num_edges()
    edge_count_each_batch = \
        torch.bincount(g.ndata['_BATCH'][indices][0], minlength=batch_size)  # Get the count of added edges for each graph in batch
    batch_num_edges += edge_count_each_batch

    # Re-assign batch information after adding edge
    g.add_edges(indices[0], indices[1], edata)
    g.set_batch_num_nodes(batch_num_nodes)
    g.set_batch_num_edges(batch_num_edges)

    return g


from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)


def get_score(targets, outputs, metric, enzymes=0):
    """
    Calculate the specified metric for the given targets and outputs.
    
    Parameters:
    targets (array-like): True labels or binary label indicators.
    outputs (array-like): Predicted labels or probabilities.
    metric (str): The metric to calculate. One of 'acc', 'prc', 'recall', 'f1', 'AUROC'.
    
    Returns:
    float: Calculated metric.
    """
    if metric == 'acc':
        return accuracy_score(targets, outputs)
    elif metric == 'prc':
        return precision_score(targets, outputs, average='micro')
    elif metric == 'recall':
        return recall_score(targets, outputs, average='weighted')
    elif metric == 'micro-f1':
        return f1_score(targets, outputs, average='micro')
    elif metric == 'macro-f1':
        return f1_score(targets, outputs, average='macro')
    elif metric == 'AUROC':
        if enzymes:
            # For multiclass classification
            return roc_auc_score(
                targets, outputs, multi_class='ovr', average='weighted'
            )
        else:
            # For binary classification
            return roc_auc_score(targets, outputs)
    else:
        raise ValueError(f"Unknown metric: {metric}")


from typing import Any, Callable, Literal, NamedTuple, Union

from scipy import optimize
from scipy.optimize import fmin


def _handle_torch_tensor(y: torch.Tensor):
    if not isinstance(y, torch.Tensor):
        return y
    return y.detach().cpu().numpy()


def find_best_binary_threshold(
    y_true_labels: np.ndarray,
    y_pred_probas: np.ndarray,
    objective_fn: Callable[[np.ndarray, np.ndarray], Any],
    Ns: int = None,  # pylint: disable=invalid-name
    bounds: tuple[float, float] = None,
    **kwargs,
) -> np.ndarray:
    """Find the best thresholds for single-label binary classification task

    Args:
        y_pred_probas (np.ndarray): predicted distribution.  shape: (n_samples,)
        y_true_labels (np.ndarray): true labels.  shape: (n_samples,)
        objective_fn  (Callable[[y_true, y_pred], Value]) sklearn-format metrics function.
            Higher is better.
        bounds (tuple[float, float]): bounds for optimization. Default to (0, 1)
        Ns: number of grid points
        kwargs: additional args pass to optimizer function `scipy.optimize.brute`.
    """

    y_pred_probas = _handle_torch_tensor(y_pred_probas)
    y_true_labels = _handle_torch_tensor(y_true_labels)

    assert len(y_pred_probas.shape) == 1, (
        f'not binary single-label due to the shape = {y_pred_probas.shape}'
    )

    def loss_fn(thresholds: np.ndarray):
        pred_labels = (y_pred_probas > thresholds).astype(int)
        return -objective_fn(y_true_labels, pred_labels)

    if Ns is not None:
        kwargs['Ns'] = Ns
    bounds = bounds or (0., 1.)
    res = optimize.brute(loss_fn, (bounds, ), **kwargs)
    return res[0]


def find_best_thresholds(
    y_true_labels: np.ndarray, y_pred_probas: np.ndarray,
    objective_fn: Union[Callable[[np.ndarray, np.ndarray], Any],
                        Literal['macro_f1', 'micro_f1']], N=1
) -> np.ndarray:
    """Find the best thresholds for multi-label binary classification task

    Args:
        y_pred_probas   (np.ndarray[n_samples, n_dims]): predicted distribution
        y_true_labels (np.ndarray[n_samples]): true labels
        objective_fn  (Callable[[y_true, y_pred], Value]) sklearn-format metrics function.
            Higher is better.
    """

    def handle_torch_tensor(y: torch.Tensor):
        if not isinstance(y, torch.Tensor):
            return y
        return y.detach().cpu().numpy()

    y_pred_probas = handle_torch_tensor(y_pred_probas)
    y_true_labels = handle_torch_tensor(y_true_labels)

    if objective_fn in ('micro_f1', 'macro_f1'):
        fn_str = objective_fn

        def f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average=fn_str[:5])

        objective_fn = f1

    def loss_fn(thresholds: np.ndarray):
        pred_labels = (y_pred_probas > thresholds).astype(int)
        return -objective_fn(y_true_labels, pred_labels)

    if N <= 0:
        raise ValueError("Number of initial points must be positive")
    elif N == 1:  # 1 initial point 0.5
        x0 = [0.5] * y_true_labels.shape[-1]
        return fmin(loss_fn, x0=x0, disp=0)
    else:
        num_rand = N
        x0 = np.random.normal(0.5, 0.1, num_rand)
        return [fmin(loss_fn, x0=x0[i], disp=0) for i in range(num_rand)]


def validate_thr(val_probs, val_targets, best_val_acc, best_thr):
    temp_val_output = to_labels(val_probs, 0.5).long().cpu().numpy()
    temp_val_acc = get_score(val_targets, temp_val_output, 'acc')
    if temp_val_acc >= best_val_acc:
        best_val_acc = temp_val_acc
        temp_thr = best_thr
    else:
        temp_thr = -1
    return best_val_acc, temp_thr


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold)
    if temp_val_acc >= best_val_acc:
        best_val_acc = temp_val_acc
        temp_thr = best_thr
    else:
        temp_thr = -1
    return best_val_acc, temp_thr


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold)
