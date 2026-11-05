"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import csv

import torch as th
import torch.nn as nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

def identity_act(input):
    return input

def get_activation(act: str, inplace=False):
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "gelu":
        return nn.GELU()
    elif act == "prelu":
        return nn.PReLU()
    elif act == "identity":
        return identity_act
    else:
        return identity_act

class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 alpha=0.):

        super(myGATConv, self).__init__()

        self._edge_feats = edge_feats
        self._num_heads = num_heads
        #self._in_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        #self.edge_emb = nn.Parameter(th.zeros(size=(num_etypes, edge_feats)))

        self.W = nn.Linear(self._in_feats, out_feats * num_heads, bias=True)
        self.W_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=True)
        #self.W = nn.Parameter(th.FloatTensor(in_feats, out_feats * num_heads))
        #self.W_e = nn.Parameter(th.FloatTensor(edge_feats, edge_feats * num_heads))
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if residual:
            self.residual = nn.Linear(in_feats, out_feats * num_heads)
        else:
            self.register_buffer("residual", None)

        self.reset_parameters()
        self.activation = None if activation is None else get_activation(activation)
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight, gain=gain)
        nn.init.xavier_normal_(self.W_e.weight, gain=gain)
    
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        #print("feat=",feat)
        #print("e_feat=",e_feat)
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            feat = self.feat_drop(feat)
            h = self.W(feat).view(-1, self._num_heads, self._out_feats)
            #h = th.matmul(feat, self.W).view(-1, self._num_heads, self._out_feats)
            #h[th.isnan(h)] = 0.0
            e_feat = self.edge_emb(e_feat)
            e_feat = self.W_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            #e = th.matmul(self.edge_emb,self.W_e).view(-1, self._num_heads, self._edge_feats)

            h_l = (self.attn_l * h).sum(dim=-1).unsqueeze(-1) # h_l = (self.a_l * h).sum(dim=-1)[row]
            h_r = (self.attn_r * h).sum(dim=-1).unsqueeze(-1) # h_r = (self.a_r * h).sum(dim=-1)[col]
            h_e = (self.attn_e * e_feat).sum(dim=-1).unsqueeze(-1) # h_e = (self.a_e * e).sum(dim=-1)[tp]
            # edge_attention = self.leakyrelu(h_l + h_r + h_e)
            graph.srcdata.update({'h': h, 'h_l': h_l})
            graph.dstdata.update({'h_r': h_r})
            graph.edata.update({'h_e': h_e})
            graph.apply_edges(fn.u_add_v('h_l', 'h_r', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('h_e'))
            # edge_attention: E * H

            # compute softmax (a = edge_attention)
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            
            # message passing
            graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                             fn.sum('m', 'h'))
            out = graph.dstdata['h']

            # residual
            if self.residual is not None:
                resval = self.residual(feat).view(feat.shape[0], -1, self._out_feats)
                out = out + resval

            # bias
            if self.bias:
                out = out + self.bias_param

            # activation
            if self.activation:
                out = self.activation(out)

            return out, graph.edata.pop('a').detach()

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self._in_feats) + " -> " + str(self._out_feats) + ")"
    
class myGATConv_hetero(nn.Module):
    def __init__(self,
                edge_feats,  # 16
                num_etypes,
                in_feats,  # 16
                out_feats,  # 16
                num_heads,  # 4
                feat_drop=0.,
                attn_drop=0.5,
                negative_slope=0.2,
                residual=False,
                activation=None,
                allow_zero_in_degree=False,
                bias=True,
                alpha=0.):

        super(myGATConv_hetero, self).__init__()

        self._edge_feats                = edge_feats
        self._num_heads                 = num_heads
        self._in_feats                  = in_feats
        self._out_feats                 = out_feats
        self._allow_zero_in_degree      = allow_zero_in_degree

        self.edge_emb = nn.Embedding(num_etypes, edge_feats)

        self.W      = nn.Linear(self._in_feats, out_feats * num_heads, bias=True)
        self.W_e    = nn.Linear(edge_feats, edge_feats * num_heads, bias=True)
        
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if residual:
            self.residual = nn.Linear(in_feats, out_feats * num_heads)
        else:
            self.register_buffer("residual", None)

        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight, gain=gain)
        nn.init.xavier_normal_(self.W_e.weight, gain=gain)
    
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        #print("feat=",feat)
        # print("\ne_feat=",e_feat)
        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                            'output for those nodes will be invalid. '
                            'This is harmful for some applications, '
                            'causing silent performance regression. '
                            'Adding self-loop on the input graph by '
                            'calling `g = dgl.add_self_loop(g)` will resolve '
                            'the issue. Setting ``allow_zero_in_degree`` '
                            'to be `True` when constructing this module will '
                            'suppress the check and let the code run.')

        feat = self.feat_drop(feat)
        h = self.W(feat).view(-1, self._num_heads, self._out_feats)
        #h = th.matmul(feat, self.W).view(-1, self._num_heads, self._out_feats)
        #h[th.isnan(h)] = 0.0
        e_feat = self.edge_emb(e_feat)
        e_feat = self.W_e(e_feat).view(-1, self._num_heads, self._edge_feats)
        #e = th.matmul(self.edge_emb,self.W_e).view(-1, self._num_heads, self._edge_feats)

        h_l = (self.attn_l * h).sum(dim=-1).unsqueeze(-1) # h_l = (self.a_l * h).sum(dim=-1)[row]
        h_r = (self.attn_r * h).sum(dim=-1).unsqueeze(-1) # h_r = (self.a_r * h).sum(dim=-1)[col]
        h_e = (self.attn_e * e_feat).sum(dim=-1).unsqueeze(-1) # h_e = (self.a_e * e).sum(dim=-1)[tp]
        # edge_attention = self.leakyrelu(h_l + h_r + h_e)
        graph.srcdata.update({'h': h, 'h_l': h_l})
        graph.dstdata.update({'h_r': h_r})
        graph.edata.update({'h_e': h_e})
        graph.apply_edges(fn.u_add_v('h_l', 'h_r', 'e'))
        e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('h_e'))
        # edge_attention: E * H

        # compute softmax (a = edge_attention)
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        if res_attn is not None:
            graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
        
        # message passing
        graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                        fn.sum('m', 'h'))
        out = graph.dstdata['h']

        # residual
        if self.residual is not None:
            resval = self.residual(feat).view(feat.shape[0], -1, self._out_feats)
            out = out + resval

        # bias
        if self.bias:
            out = out + self.bias_param

        # activation
        if self.activation:
            out = self.activation(out)

        graph.dstdata['h'] = out
        
        return graph, out

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self._in_feats) + " -> " + str(self._out_feats) + ")"