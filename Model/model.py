from Utils.mol_utils import ATOM_IDX, MAX_SEQ_LENGTH, VOCABULARY

import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgllife.model.gnn.mpnn import MPNNGNN
from dgl.nn.pytorch import Set2Set, SumPooling


class FeatureExtractor(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats=64, embedding_dim_node=16, embedding_dim_edge=32):
        super().__init__()
        self.node_emb = nn.Embedding(num_embeddings=node_in_feats, embedding_dim=embedding_dim_node)
        self.edge_emb = nn.Embedding(num_embeddings=edge_in_feats, embedding_dim=embedding_dim_edge)
        self.mpnn = MPNNGNN(node_in_feats=embedding_dim_node, edge_in_feats=embedding_dim_edge,
                            node_out_feats=hidden_feats)
        # self.pool = Set2Set(input_dim=hidden_feats, n_iters=2, n_layers=1)
        self.pool = SumPooling()

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor, edge_feat: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        h_node = self.node_emb(node_feat)
        h_edge = self.edge_emb(edge_feat)
        h_node = self.mpnn(g, h_node, h_edge)
        h_graph = self.pool(g, h_node)

        return h_graph, h_node


class NodeHeader(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeHeader, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.fc2 = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class EdgeRNNHeader(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, h_graph_dim, input_size_rnn, hidden_size_rnn, num_layers,
                 hidden_size_head, out_size_head, seq_len):
        super(EdgeRNNHeader, self).__init__()
        self.atom_emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.state_translator = nn.Linear(in_features=embedding_dim+h_graph_dim, out_features=hidden_size_rnn)
        self.rnn = nn.GRU(input_size=input_size_rnn, hidden_size=hidden_size_rnn, num_layers=num_layers, batch_first=True)
        self.header = nn.Sequential(
            nn.Linear(in_features=hidden_size_rnn, out_features=hidden_size_head),
            nn.Linear(in_features=hidden_size_head, out_features=out_size_head)
        )
        self.x_len = seq_len

    def forward(self, x, x_len, h_graph, x_atom):
        # Prepare an Embedded Representation of Atoms and Graphs
        emb_atom = self.atom_emb(x_atom)
        state = self.state_translator(torch.cat([h_graph, emb_atom], dim=1))
        state = torch.unsqueeze(state, 0)
        state = state.expand(self.rnn.num_layers, state.shape[1], state.shape[2])

        # RNN
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x, state)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.x_len)

        h = x.contiguous().view(-1, x.shape[2])
        y = self.header(h)
        y = y.contiguous().view(x.shape[0], x.shape[1], y.shape[1])

        return y



