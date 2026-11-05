import torch
import numpy as np

def get_batch_num_nodes(g, ntypes):
    # Batch num nodes of each ntype
    batch_num_nodes_obj = []
    for nid, ntype in enumerate(ntypes):
        batch_num_nodes_type    = torch.tensor([], dtype=torch.long, device=g.device)
        non_zero_indices_ntype  = torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
        batch_by_ntype          = g.ndata['_BATCH'][non_zero_indices_ntype]
        for bid in range(g.batch_size):
            count = (batch_by_ntype == bid).sum()
            batch_num_nodes_type = torch.cat([batch_num_nodes_type, count.unsqueeze(0)])
        batch_num_nodes_obj.append(batch_num_nodes_type)
    return batch_num_nodes_obj
