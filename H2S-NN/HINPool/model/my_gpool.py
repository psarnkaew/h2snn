import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.conv import myGATConv, myGATConv_hetero
from model.model_utils import *
import dgl
from utils.utils import check_zero_degree, add_nhop_edges_in_layer
import time

class HGPool(nn.Module):
    # def __init__(self, in_dim, n_classes, args):
    def __init__(self, args, G, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGPool, self).__init__()
        self.args           = args
        self.sample_graph   = G
        self.n_hid          = n_hid
        self.act            = nn.ReLU()
        self.feat_fusion    = args.fusion
        self.skip_aggr      = args.skip_aggr
        self.softmax        = nn.Softmax(dim=0)

        # Encoder + Pooling
        self.backbone   = args.backbone
        if args.dataset not in ['NCI1', 'NCI109']:
            self.feat_proj  = nn.ModuleList()
            for ntype in G.ntypes:
                self.feat_proj.append(nn.Linear(n_inp, n_hid))
        else:
            self.feat_proj  = nn.Linear(n_inp, n_hid)
        self.mypool     = myPool(
            G, args, args.backbone, args.pool_ratio, n_hid, n_out, n_heads, self.act, args.dropout, use_norm)
        
        # Cross-layer Aggregation by Attention part
        if self.args.do_cross_layer:
            pass
        else:
            cross_layer_att_dim = None
            if self.args.do_TA_readout:
                cross_layer_att_dim = n_hid * n_heads
            else:
                if self.args.use_mixed:
                    if self.feat_fusion == 'concat':
                        cross_layer_att_dim = (len(G.ntypes)+1) * n_hid * n_heads
                    elif self.feat_fusion in ['mean', 'max', 'att_shared', 'att_sep', 'self_att_shared', 'self_att_sep']:
                        cross_layer_att_dim = 2 * n_hid * n_heads
                else:
                    if self.feat_fusion == 'concat':
                        cross_layer_att_dim = len(G.ntypes) * n_hid * n_heads
                    elif self.feat_fusion in ['mean', 'max', 'att_shared', 'att_sep', 'self_att_shared', 'self_att_sep']:
                        cross_layer_att_dim = n_hid * n_heads
            self.cross_layer_att = nn.Parameter(torch.randn(len(args.pool_ratio) + 1, cross_layer_att_dim))
        
        # Feature Fusion by Attention part
        if self.feat_fusion == 'concat':
            self.layer_norm = nn.LayerNorm(len(G.ntypes) * n_hid * n_heads)
        elif self.feat_fusion in ['mean', 'max', 'att_shared', 'att_sep', 'self_att_shared', 'self_att_sep']:
            self.layer_norm = nn.LayerNorm(n_hid * n_heads)
        
        if self.feat_fusion == 'att_shared':
            self.feat_att_shared = nn.Parameter(torch.randn(len(G.ntypes), n_hid * n_heads))
        # self.feat_att_shared = nn.Embedding(len(G.ntypes), n_hid * n_heads)
        elif self.feat_fusion in ['att_sep'] :
            self.feat_att_sep    = nn.Parameter(torch.randn(len(args.pool_ratio) + 1, \
                                                            len(G.ntypes), n_hid * n_heads))
        # self.attention_weights = nn.Parameter(torch.Tensor(n_hid * n_heads, n_hid * n_heads))
        elif self.feat_fusion == 'self_att_shared':
            self.q_shared = nn.Linear(n_hid * n_heads, n_hid * n_heads)
            self.k_shared = nn.Linear(n_hid * n_heads, n_hid * n_heads)
            self.v_shared = nn.Linear(n_hid * n_heads, n_hid * n_heads)
            self.scale = 1.0 / ((n_hid * n_heads) ** 0.5)
        elif self.feat_fusion == 'self_att_sep':
            self.q_seps   = nn.ModuleList()
            self.k_seps   = nn.ModuleList()
            self.v_seps   = nn.ModuleList()
            for _ in range(len(args.pool_ratio) + 1):
                self.q_seps.append(nn.Linear(n_hid * n_heads, n_hid * n_heads))
                self.k_seps.append(nn.Linear(n_hid * n_heads, n_hid * n_heads))
                self.v_seps.append(nn.Linear(n_hid * n_heads, n_hid * n_heads))
            self.scale = 1.0 / ((n_hid * n_heads) ** 0.5)
                            

        # Classify part
        # layer_num: The factor of input embedding dimension of classification layer
        if not self.args.do_cross_layer:
            if self.skip_aggr in ['concat']:
                layer_num = len(args.pool_ratio) + 1
            elif self.skip_aggr in ['mean', 'max', 'att']:
                layer_num = 1
        else:
            layer_num = 1

        # out_l_full: Final classification layer
        out_l_dim = None
        mixed_scale = self.args.use_mixed + 1
        if not self.args.do_TA_readout:
            if self.feat_fusion in ['concat']:
                if mixed_scale == 1:
                    out_l_dim = mixed_scale * layer_num * len(G.ntypes) * n_hid * n_heads
                else:
                    out_l_dim = layer_num * (len(G.ntypes)+1) * n_hid * n_heads
            elif self.feat_fusion in ['mean', 'max', 'att_shared', 'att_sep', 'self_att_shared', 'self_att_sep']:
                out_l_dim = mixed_scale * layer_num * n_hid * n_heads
        else:
            out_l_dim = layer_num * n_hid * n_heads
        self.out_l_full = nn.Linear(out_l_dim, n_hid)
        self.out_l_2    = nn.Linear(n_hid, n_out)
        
        self.out_drop   = nn.Dropout(p=args.dropout)
        Initializer.weights_init(self)

    def forward(self, graph):
        h = torch.zeros((graph.num_nodes(), self.n_hid))
        if self.args.dataset not in ['NCI1', 'NCI109']:
            for n_id, ntype in enumerate(self.sample_graph.ntypes):
                non_zero_indices = \
                    torch.nonzero(graph.ndata['_TYPE']==n_id)
                h[non_zero_indices] = torch.tanh(self.feat_proj[n_id](graph.ndata['feat'][non_zero_indices]))
        else:
            h = torch.tanh(self.feat_proj(graph.ndata['feat']))
        graph.ndata['h'] = h
        ###############################
        graphs, batch_nums = self.mypool(graph, self.args.pool_by)

        out_full = self.readout_full_Simple(graphs, batch_nums)
        logits = self.classify_full(out_full)
        return logits

    def readout_full(self, gs):
        h = torch.tensor([])
        if self.feat_fusion == 'concat':
            for g in gs:
                for ntype in g.ntypes:
                    mean_emb = dgl.readout_nodes(g, 'h', op='mean', ntype=ntype)
                    h = torch.cat((h, torch.nan_to_num(mean_emb)), dim=1)
        elif self.feat_fusion == 'mean':
            hs = {}
            for gid, g in enumerate(gs):
                for ntype in g.ntypes:
                    mean_emb = dgl.readout_nodes(g, 'h', op='mean', ntype=ntype)
                    if gid not in hs.keys():
                        hs[gid] = torch.nan_to_num(mean_emb)
                    else:
                        hs[gid] += torch.nan_to_num(mean_emb)
            for gid, embedding in hs.items():
                h = torch.cat((h, embedding/len(g.ntypes)), dim=1)
        elif self.feat_fusion == 'max':
            hs = {}
            for g in gs:
                for ntype in g.ntypes:
                    mean_emb = dgl.readout_nodes(g, 'h', op='mean', ntype=ntype)
                    if ntype not in hs.keys():
                        hs[ntype] = torch.nan_to_num(mean_emb)
                    else:
                        hs[ntype] += torch.nan_to_num(mean_emb)
            stacked_h = torch.stack(tuple(hs.values()))
            h = torch.max(stacked_h, dim=0)[0]
        # elif self.feat_fusion == 'att': # 擱置
        #     # TODO
        #     hs = ()
        #     for g_idx, g in enumerate(gs):
        #         h_per_type = ()
        #         for ntype in g.ntypes:
        #             mean_emb = dgl.readout_nodes(g, 'h', op='mean', ntype=ntype)
        #             h_per_type += (mean_emb,)
        #         h_per_type = torch.stack(h_per_type)

        #         intra_score = self.softmax(torch.matmul(h_per_type, self.intra_feat_att))
        #         print(intra_score)
        #         h_per_graph = torch.sum(intra_score.unsqueeze(-1) * h_per_type, dim=0)
        #         hs += (h_per_graph,)
        #     hs = torch.stack(hs)
        #     # print(hs.shape, self.inter_feat_att.shape)
        #     inter_score = self.softmax(torch.matmul(hs, self.inter_feat_att))
        #     print(inter_score)
        #     # print(self.softmax(torch.matmul(hs, self.inter_feat_att)))
        #     # print(inter_score.shape)
        #     print(hs)
        
        return h
    def readout_full_Simple(self, gs, batch_nums):
        h = torch.tensor([])
        hs = {}
        if not self.args.do_TA_readout:
            if self.feat_fusion == 'concat':
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')
                        if gid not in hs.keys():
                            hs[gid] = torch.nan_to_num(mean_emb)
                        else:
                            hs[gid] = torch.cat([hs[gid], torch.nan_to_num(mean_emb)], dim=1)

            elif self.feat_fusion == 'mean':
                # EA of readouts of ntypes
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')
                        if gid not in hs.keys():
                            hs[gid] = torch.nan_to_num(mean_emb)
                        else:
                            hs[gid] += torch.nan_to_num(mean_emb)
                    hs[gid] /= len(self.sample_graph.ntypes)

            elif self.feat_fusion == 'max':
                # EM of readouts of ntypes
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')
                        if gid not in hs.keys():
                            hs[gid] = torch.nan_to_num(mean_emb).unsqueeze(-1)
                        else:
                            hs[gid] = torch.cat([hs[gid], torch.nan_to_num(mean_emb).unsqueeze(-1)], dim=2)
                    hs[gid] = torch.max(hs[gid], dim=-1)[0]

            elif self.feat_fusion == 'att_shared':
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')

                        if gid not in hs.keys():
                            hs[gid] = torch.nan_to_num(mean_emb * self.feat_att_shared[nid])
                        else:
                            hs[gid] += torch.nan_to_num(mean_emb * self.feat_att_shared[nid])
            
            elif self.feat_fusion == 'att_sep':
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        if non_zero_indices_ntype[0].shape[0] == 0:
                            continue
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')

                        if gid not in hs.keys():
                            hs[gid]  = mean_emb * self.feat_att_sep[gid, nid]
                        else:
                            hs[gid] += mean_emb * self.feat_att_sep[gid, nid]

            elif self.feat_fusion == 'self_att_shared':
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    embeds = []
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')
                        embeds.append(mean_emb)
                    stacked_embed = torch.stack(embeds, dim=1)
                    
                    q_emb = self.q_shared(stacked_embed)
                    k_emb = self.k_shared(stacked_embed)
                    v_emb = self.v_shared(stacked_embed)

                    att_scores = torch.matmul(q_emb, k_emb.transpose(-2, -1)) 
                    att_scores = att_scores * self.scale
                    att_weights = F.softmax(att_scores, dim=-1)

                    fused_emb = torch.matmul(att_weights, v_emb) 
                    fused_emb = fused_emb.sum(dim=1) 

                    hs[gid]  = fused_emb
            
            elif self.feat_fusion == 'self_att_sep':
                for gid, (g, batch_num) in enumerate(zip(gs, batch_nums)):
                    embeds = []
                    for nid, ntype in enumerate(self.sample_graph.ntypes):
                        non_zero_indices_ntype = \
                            torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
                        h_by_type   = g.ndata['h'][non_zero_indices_ntype].flatten(1)
                        mean_emb    = dgl.ops.segment_reduce(batch_num[nid], h_by_type, 'mean')
                        embeds.append(mean_emb)
                    stacked_embed = torch.stack(embeds, dim=1)
                    
                    q_emb = self.q_seps[gid](stacked_embed)
                    k_emb = self.k_seps[gid](stacked_embed)
                    v_emb = self.v_seps[gid](stacked_embed)

                    att_scores = torch.matmul(q_emb, k_emb.transpose(-2, -1)) 
                    att_scores = att_scores * self.scale
                    att_weights = F.softmax(att_scores, dim=-1)

                    fused_emb = torch.matmul(att_weights, v_emb)
                    fused_emb = fused_emb.sum(dim=1) 

                    hs[gid]  = fused_emb
        
            # LayerNorm first
            if self.args.norm:
                for gid, g in enumerate(gs):
                    hs[gid] = self.layer_norm(hs[gid])
            # Concat mixed readout
            if self.args.use_mixed:
                for gid, g in enumerate(gs):
                    hs[gid] = torch.cat([hs[gid], dgl.readout_nodes(g, 'h').flatten(1)], 1)
        else:
            for gid, g in enumerate(gs):
                hs[gid] = dgl.readout_nodes(g, 'h').flatten(1)

        # Skip-layer conncetion part
        if not self.args.do_cross_layer:
            if self.skip_aggr == 'concat':
                h = torch.cat(list(hs.values()), dim=1)
            elif self.skip_aggr == 'mean':
                h = torch.sum(torch.stack(list(hs.values())), dim=0) / len(hs.keys())
            elif self.skip_aggr == 'max':
                h = torch.max(torch.stack(list(hs.values())), dim=0)[0]
            elif self.skip_aggr == 'att':
                h = torch.stack(list(hs.values())) * self.cross_layer_att.unsqueeze(1)
                h = torch.sum(h, dim=0)
        else:
            h = torch.tensor(list(hs.values())[-1])
            
        return h
    def classify_full(self, h):
        # h = self.out_drop(h)
        h = self.out_l_full(h.flatten(1))
        h = self.act(h)
        # h = self.out_drop(h)
        h = self.out_l_2(h)
        # return F.log_softmax(h, dim=1)
        return h
    
    def readout(self, g):
        h = torch.tensor([])
        for ntype in g.ntypes:
            mean_emb = dgl.readout_nodes(g, 'h', op='mean', ntype=ntype)
            h = torch.cat((h, torch.nan_to_num(mean_emb)), dim=1)
        return h
    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels):
        loss = nn.CrossEntropyLoss()(logits, labels.flatten())
        # loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return acc, loss

class myPool(nn.Module):
    def __init__(self, G, args, backbone, pool_ratio, n_hid, n_out, n_heads, act, dropout, use_norm):
        super(myPool, self).__init__()
        self.sample     = G
        self.args       = args
        self.pool_ratio = pool_ratio
        self.backbone   = backbone
        self.down       = nn.ModuleList()
        self.up         = nn.ModuleList()
        self.pools      = nn.ModuleList()
        self.unpools    = nn.ModuleList()
        self.l_n        = len(pool_ratio)
        self.num_etypes = len(G.etypes) if args.dataset not in ['NCI1', 'NCI109']\
                    else  len(G.etypes)

        # 1st layer, input projection
        self.down.append(myGATConv_hetero(
            n_hid,
            self.num_etypes,
            n_hid,
            n_hid,
            n_heads,
            dropout,
            dropout,
            negative_slope=0.5,
            residual=True,
            activation=nn.ReLU(),
            allow_zero_in_degree=True,
            alpha=0.05,
            )
        )
        self.pools.append(Pool(G, args, pool_ratio[0], n_hid, dropout))
        # 2nd ~ last layer
        for i in range(1, self.l_n):
            self.down.append(myGATConv_hetero(
                n_hid,
                self.num_etypes,
                n_hid * n_heads,
                n_hid,
                n_heads,
                dropout,
                dropout,
                negative_slope=0.5,
                residual=True,
                activation=nn.ReLU(),
                allow_zero_in_degree=True,
                alpha=0.05,
                )
            )
            self.pools.append(Pool(G, args, pool_ratio[i], n_hid, dropout))
        # Final encoder
        self.bottom = myGATConv_hetero(
            n_hid,
            self.num_etypes,
            n_hid * n_heads,
            n_hid,
            n_heads,
            dropout,
            dropout,
            negative_slope=0.5,
            residual=True,
            activation=nn.ReLU(),
            allow_zero_in_degree=True,
            alpha=0.05,
        )

    def forward(self, g, pool_by='type'):
        gs          = []
        batch_nums  = []
        
        res_attn = None
        hg = self.sample

        ### Inject batch id due to the early transformation into homo graph ###
        gid_offset = 8192  # Avoid replacement conflict
        # Quick replacement of graph id
        boundaries_gid  = torch.diff(g.ndata['gid'], prepend=torch.tensor([float('nan')])) != 0
        values_gid      = g.ndata['gid'][boundaries_gid]
        gids            = {values_gid[i].item():i+gid_offset for i in range(len(values_gid))}
        for value, replace in gids.items():
            g.ndata['gid'][g.ndata['gid'] == value] = replace
        
        # Obtain num nodes graph by graph
        result_dict = {gid: \
                        torch.zeros(len(hg.ntypes), dtype=torch.long) for gid in range(g.batch_size)}
        for gid in range(g.batch_size):
            # Get boundaries where nids change and the corresponding nids
            boundaries = torch.diff(g.ndata[dgl.NTYPE][g.ndata['gid'] == gid + gid_offset], \
                                        prepend=torch.tensor([float('nan')])) != 0
            nids = g.ndata[dgl.NTYPE][g.ndata['gid'] == gid + gid_offset][boundaries]

            # Calculate the nums of  each value
            nums = torch.diff(torch.cat(\
                [torch.nonzero(boundaries).flatten(), torch.tensor([len(g.ndata[dgl.NTYPE][g.ndata['gid'] == gid + gid_offset])])]))
            
            # Populate the dictionary with the nums
            for value, num in zip(nids, nums):
                result_dict[gid][value.item()] = num.item()
        batch_id = torch.tensor([], dtype=torch.long)
        for gid in range(g.batch_size):
            temp = result_dict[gid]
            for nid, ntype in enumerate(hg.ntypes):
                batch_id = torch.cat([batch_id, torch.full((temp[nid],), gid)])
        g.ndata['_BATCH'] = batch_id
        ################################################################################
        for i in range(self.l_n):
            # e_feat: concat id of all etype in dim 1
            h = g.ndata['h'].flatten(1)
            g, h = self.down[i](g, h, g.edata['id'], res_attn=res_attn)
            gs.append(g.clone())
            g = add_nhop_edges_in_layer(g, self.args.hop)
            g, orig_batch_num_nodes = self.pools[i](g, pool_by)
            batch_nums.append(orig_batch_num_nodes)
            res_attn = g.edata.pop('a').detach()
        h = g.ndata['h'].flatten(1)
        g, h = self.bottom(g, h, g.edata['id'], res_attn=res_attn)
        batch_nums.append(get_batch_num_nodes(g, hg.ntypes))

        gs.append(g.clone())

        return gs, batch_nums

class Pool(nn.Module):
    # Return pooled graph and index after pooling
    # Since the embedding is saved in graph.ndata, no need to return
    def __init__(self, G, args, k, in_dim, p):
        super(Pool, self).__init__()
        self.sample_graph   = G
        self.args           = args
        self.mlp_score      = args.mlp_score
        self.multiply       = args.mul_score
        self.backbone       = args.backbone
        self.k              = k

        self.sigmoid    = nn.Sigmoid()
        self.proj       = nn.ModuleDict()
        self.drop       = nn.Dropout(p=p) if p > 0 else nn.Identity()

        if not self.mlp_score:
            for nid, ntype in enumerate(G.ntypes):
                self.proj[ntype] = nn.Linear(in_dim * args.num_heads, 1)
        else: 
            for nid, ntype in enumerate(G.ntypes):
                self.proj[ntype] = MLP_score(in_dim * args.num_heads, in_dim * args.num_heads, 1)

    def forward(self, g, pool_by):
        g.ndata['score'] = torch.zeros((len(g.nodes())))
        for nid, ntype in enumerate(self.sample_graph.ntypes):
            non_zero_indices = \
                torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
            h = g.ndata['h'][non_zero_indices].flatten(1)
            weights = self.proj[ntype](h)
            g.ndata['score'][non_zero_indices] = self.sigmoid(weights).squeeze()
            
        return top_k_graph_simple(g, self.k, self.sample_graph.ntypes, pool_by, self.multiply)
            
class MLP_score(nn.Module):
    # 2 layer MLP for temp
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(MLP_score, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.out(x)
        
        return x

def top_k_graph_simple(g, k, ntypes, pool_by, multiply):

    # Batch num nodes of each ntype
    batch_num_nodes_obj = get_batch_num_nodes(g, ntypes)

    # Type-Aware TopK
    if pool_by == 'type':
        # Refresh number of nodes of each node type everytime after dropping
        num_nodes_by_ntype = torch.tensor(\
            [len(torch.nonzero(g.ndata['_TYPE']==nid)) for nid in range(len(ntypes))])
        cum_nums_by_ntype   = torch.cumsum(num_nodes_by_ntype, 0)
        start_idx_by_ntype  = torch.cat((torch.tensor([0]), cum_nums_by_ntype))
        save_index_all      = torch.tensor([], dtype=torch.long)


        # ======================================================== # 
        for nid, ntype in enumerate(ntypes):
            non_zero_indices = \
                    torch.nonzero(g.ndata['_TYPE']==nid, as_tuple=True)
            num_nodes = len(non_zero_indices[0])
            if num_nodes == 0:
                continue
            
            save_idx = batch_topk_simple(g.clone(), 'score', \
                torch.ceil(k*batch_num_nodes_obj[nid]).to(torch.long), batch_num_nodes_obj[nid], \
                        pool_by, non_zero_indices)
            
            shift = 0
            start_ids = []
            for i in range(len(batch_num_nodes_obj[nid])):
                start_ids.append(shift)
                shift += batch_num_nodes_obj[nid][i].item()

            save = torch.cat([save_idx[i]+start_ids[i] for i in range(len(save_idx))]) + start_idx_by_ntype[nid]
            save_index_all = torch.cat([save_index_all, save])

        # Inverse save indices to get drop indices then drop
        # drop_idx = torch.tensor(\
        #     list(set(torch.arange(g.num_nodes()).tolist()) - set(save_index_all.tolist())) \
        #     , dtype=torch.long)
        # Inverse save indices to get drop indices then drop
        # start_2 = torch.cuda.Event(enable_timing=True)
        # end_2 = torch.cuda.Event(enable_timing=True)

        # start_2.record()
        drop_idx_temp, counts = torch.unique(torch.cat([torch.arange(g.num_nodes()), save_index_all]), \
                    sorted=False, return_counts=True)
        # end_2.record()
        # torch.cuda.synchronize()
        # print(start_2.elapsed_time(end_2))
        
        drop_idx = drop_idx_temp[counts == 1]
        
    elif pool_by == 'all':
        """
        save_idx: List of nodes to be saved according to node type
        """
        save_idx = batch_topk_simple(g.clone(), 'score', k, g.batch_num_nodes(), pool_by)
        save = set(torch.cat(save_idx).tolist())

        all_id = set(torch.arange(0, g.num_nodes()).tolist())
        drop_idx = torch.tensor(list(all_id - save), dtype=torch.long)
    
    g.remove_nodes(drop_idx)
    # Multiply by score after dropping all nodes
    if multiply:
        g.ndata['h'] = g.ndata['h'] * g.ndata['score'].view(-1, 1, 1)
    
    return g, batch_num_nodes_obj

def batch_topk_simple(graph, feat_name, k, batch_num_nodes, \
                      pool_by='type', non_zero_indices=None, descending=True):
    '''
    Args 
        graph:
            The graph ids are placed due to the order of subgraphs
        k: 
            Type int if pool by type -- Actual num of nodes to be saved
            Type float if pool by all -- The ratio of nodes to be saved
        non_zero_indices:
            Only given when pool by type.
            None when pool by all.
    '''
    batch_k = k if pool_by == 'type' \
        else torch.ceil(k * batch_num_nodes).to(torch.long)
    feat    = graph.ndata[feat_name][non_zero_indices] if pool_by == 'type' \
        else graph.ndata[feat_name]

    lengths         = batch_num_nodes
    cum_lengths     = torch.cumsum(lengths, 0)
    start_idx       = torch.cat((torch.tensor([0]), cum_lengths))
    feats           = [feat[start_idx[i]:start_idx[i+1]] for i in range(len(batch_k))]
    topk_indices    = [torch.sort(temp, dim=0, descending=descending)[1] for temp in feats]
    
    if pool_by == 'type':
        return [topk_indices[i][:batch_k[i]] for i in range(len(lengths))]
    elif pool_by == 'all':
        return [topk_indices[i][:batch_k[i]] + start_idx[i] for i in range(graph.batch_size)]

def top_k_graph(g, k, pool_by):
    # Type-Aware TopK
    if pool_by == 'type':
        for nid, ntype in enumerate(g.ntypes):
            # 0, 1, 2
            if g.num_nodes(ntype) == 0:
                continue
            num_nodes = g.batch_num_nodes(ntype)
            
            save_idx = batch_topk(g.clone(), 'score', torch.ceil(k*num_nodes).to(torch.long), pool_by, ntype)
            
            shift = 0
            start_ids = []
            for i in range(len(num_nodes)):
                start_ids.append(shift)
                shift += num_nodes[i].item()
            save = torch.cat([save_idx[i]+start_ids[i] for i in range(len(save_idx))])
            # print(save)
            drop_idx = torch.tensor(list(set(torch.arange(0, g.num_nodes(ntype)).tolist()) - set(save.tolist())), dtype=torch.long)
            
            g.remove_nodes(drop_idx, ntype=ntype)
            g.nodes[ntype].data['h'] = torch.mul(g.ndata['h'][ntype], g.ndata['score'][ntype])
        return g
    elif pool_by == 'all':
        """
        save_idx: Dictionary of nodes to be save according to node type
        """
        save_idx = batch_topk(g.clone(), 'score', k, pool_by)

        # print(save_idx)
        for ntype in g.ntypes:
            all_id = set(torch.arange(0, g.num_nodes(ntype)).tolist())
            save = set(save_idx[ntype].tolist())

            drop_idx = torch.tensor(list(all_id-save), dtype=torch.long)
            g.remove_nodes(drop_idx, ntype=ntype)
            g.nodes[ntype].data['h'] = torch.mul(g.ndata['h'][ntype], g.ndata['score'][ntype])
        return g

def batch_topk(graph, feat_name, k, pool_by='type', ntype=None, descending=True):
    '''
    Argument k: 
        Type int if pool by type -- Actual num of nodes to be saved
        Type float if pool by all -- The ratio of nodes to be saved
    '''
    READOUT_ON_ATTRS = {
        "nodes": ("ndata", "batch_num_nodes", "number_of_nodes"),
        "edges": ("edata", "batch_num_edges", "number_of_edges"),
    }
    batch_k                     = k
    typestr                     = 'nodes'
    _, batch_num_objs_attr, _   = READOUT_ON_ATTRS[typestr]

    if pool_by == 'type':
        data = getattr(graph, typestr)[ntype].data
        feat = data[feat_name]

        lengths = getattr(graph, batch_num_objs_attr)(ntype)

        cum_lengths     = torch.cumsum(lengths, 0)
        start_idx       = torch.cat((torch.tensor([0]), cum_lengths))
        feats           = [feat[start_idx[i]:start_idx[i+1]] for i in range(len(batch_k))]
        topk_indices    = [torch.sort(temp, dim=0, descending=descending)[1] for temp in feats]
        
        return [topk_indices[i][:batch_k[i], -1] for i in range(len(lengths))]
    elif pool_by == 'all':
        # TODO: Modification or Total destruction
        # print(f'\nSaving {k} nodes')
        scores = torch.tensor([])
        lengths = torch.tensor([])
        for ntype in graph.ntypes:
            data = getattr(graph, typestr)[ntype].data
            feat = data[feat_name]
            scores = torch.cat((scores, feat))
            batch_num_node = getattr(graph, batch_num_objs_attr)(ntype).view(1, -1)
            lengths = torch.cat((lengths, batch_num_node), dim=0)
        scores = torch.cat((scores, torch.tensor([[-1]])))
        
        num_nodes_discrete = lengths
        num_nodes_by_batch = torch.sum(lengths, dim=0)
        num_nodes_by_ntype = torch.sum(lengths, dim=1)
        batch_k = torch.ceil(batch_k*num_nodes_by_batch).to(torch.long)
        # print(f'num_nodes_discrete\n{num_nodes_discrete}')
        # print(f'num_nodes_by_batch\n{num_nodes_by_batch}')
        # print(f'num_nodes_by_ntype\n{num_nodes_by_ntype}')
        # print(f'batch_k\n{batch_k}')

        shift = 0
        start_ids_by_ntype = []
        for i in range(len(num_nodes_by_ntype)):
            start_ids_by_ntype.append(shift)
            shift += num_nodes_by_ntype[i].item()
            
        cum_lengths_by_ntype = torch.cumsum(num_nodes_discrete, 1)
        cum_lengths_by_ntype = torch.cat((torch.zeros((len(graph.ntypes), 1)), cum_lengths_by_ntype), dim=1).to(torch.long)
        start_ids_by_ntype = (cum_lengths_by_ntype + torch.tensor(start_ids_by_ntype, dtype=torch.long).unsqueeze(1))
        # print(f'start_ids_by_ntype\n{start_ids_by_ntype}')
        feats = [torch.cat([scores[start_ids_by_ntype[nid, gid]:start_ids_by_ntype[nid, gid+1], :] for nid in range(len(graph.ntypes))]) for gid in range(len(batch_k))]
        sorted_feats = [torch.topk(temp, batch_k[gid], dim=0)[1] for gid, temp in enumerate(feats)]
        topk_indices = sorted_feats
        ############################################################################################################################
        shift = 0
        start_ids_by_graph = []
        for i in range(len(num_nodes_by_batch)):
            start_ids_by_graph.append(shift)
            shift += num_nodes_by_batch[i].item()
        cum_lengths_by_graph = torch.cumsum(num_nodes_discrete, 0)
        cum_lengths_by_graph = torch.cat((torch.zeros((1, len(num_nodes_by_batch))), cum_lengths_by_graph), dim=0)
        start_ids_by_graph = cum_lengths_by_graph + torch.tensor(start_ids_by_graph).unsqueeze(0)
        start_ids_by_graph = start_ids_by_graph.to(torch.long)
        # print(f'start_ids_by_graph\n{start_ids_by_graph}')
        ############################################################################################################################
        save_ids = {ntype: torch.tensor([], dtype=torch.long) for ntype in graph.ntypes}
        # print(f'cum_lengths_by_ntype\n{cum_lengths_by_ntype}')
        for gid, topk_index in enumerate(topk_indices):
            # print(gid, topk_index)
            topk_index += start_ids_by_graph[0, gid]
            start = start_ids_by_graph[:, gid]
            # print(topk_index)
            # print(f'start: {start}')
            for nid, ntype in enumerate(graph.ntypes):
                mask = (topk_index >= start[nid]) & (topk_index < start[nid+1])
                partition = topk_index[mask] - start_ids_by_graph[nid, gid] + cum_lengths_by_ntype[nid, gid]
                # print(start_ids_by_graph[nid, gid])
                # print(cum_lengths_by_ntype[nid, gid])
                # print(nid, partition)
                save_ids[ntype] = torch.cat([save_ids[ntype], partition])
                
        return save_ids

class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
