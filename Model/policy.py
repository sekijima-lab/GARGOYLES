from copy import deepcopy
import math
import time
from scipy import sparse
import scipy.signal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import gym
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem

# from layers.gin_e_layer import *
from utils import ATOM_IDX, BOND_IDX, MAX_NODE, device



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# DGL operations
msg = fn.copy_src(src='x', out='m')


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}


def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        h_in = g.ndata['x']
        if self.agg == "sum":
            g.update_all(msg, reduce_sum)
        elif self.agg == "mean":
            g.update_all(msg, reduce_mean)
        h = self.linear1(g.ndata['x'])
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        # h = self.dropout(h)
        if self.residual:
            h += h_in
        return h


class GCNEmbed(nn.Module):
    def __init__(self, cfg):

        ### GCN
        super().__init__()
        self.possible_atoms = ATOM_IDX
        self.bond_type_num = cfg["emb"]["bond_type_num"]
        self.d_n = cfg["emb"]["feat_dim"]

        self.emb_size = cfg["emb"]["emb_dim"]

        in_channels = 16
        self.emb_linear = nn.Embedding(num_embeddings=len(ATOM_IDX), embedding_dim=in_channels)

        self.gcn_layers = nn.ModuleList([GCN(in_channels, self.emb_size, agg="sum", residual=False)])
        for _ in range(cfg["emb"]["num_layers"] - 1):
            self.gcn_layers.append(GCN(self.emb_size, self.emb_size, agg="sum"))

        self.pool = SumPooling()

    def forward(self, ob):
        ## Graph
        ob_g = [o['g'] for o in ob]
        # ob_att = [o['att'] for o in ob]
        #
        # # create attachment point mask as one-hot
        # for i, x_g in enumerate(ob_g):
        #     att_onehot = F.one_hot(torch.LongTensor(ob_att[i]),
        #                            num_classes=x_g.number_of_nodes()).sum(0)
        #     ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(device)

        g.ndata['x'] = self.emb_linear(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h

        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])

        return g, emb_node, emb_graph


class GCNActorCritic(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        # build policy and value functions
        self.embed = GCNEmbed(cfg)
        ob_space = env.observation_space
        ac_space = env.action_space
        self.env = env
        self.pi = SFSPolicy(ob_space, ac_space, env, cfg)
        self.v = GCNVFunction(ac_space, cfg)

    def step(self, o_g_emb, o_n_emb, o_g):
        with torch.no_grad():

            ac, ac_prob, log_ac_prob = self.pi(o_g_emb, o_n_emb, o_g)
            ac["first"] = ac["first"].cpu()
            ac["second"] = ac["second"].cpu()
            ac["third"] = ac["third"].cpu()

            dists = self.pi._distribution(ac_prob)
            logp_a = self.pi._log_prob_from_distribution(dists, ac.cpu())

            v = self.v(o_g_emb)

            ac["first"] = ac["first"].cpu().numpy()
            ac["second"] = ac["second"].cpu().numpy()
            ac["third"] = ac["third"].cpu().numpy()
            logp_a["first"] = logp_a["first"].cpu().numpy()
            logp_a["second"] = logp_a["second"].cpu().numpy()
            logp_a["third"] = logp_a["third"].cpu().numpy()

        return ac, v.cpu().numpy(), logp_a

    def act(self, o_g_emb, o_n_emb, o_g):
        return self.step(o_g_emb, o_n_emb, o_g)[0]


from scipy.special import kl_div


class GCNVFunction(nn.Module):
    def __init__(self, ac_space, cfg, override_seed=False):
        super().__init__()
        if override_seed:
            seed = cfg.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = cfg.value.batch_size
        self.device = device
        self.emb_size = cfg.value.emb_dim
        self.max_action_stop = 2

        self.d = 2 * cfg.value.emb_dim
        self.out_dim = 1

        self.embed = GCNEmbed(cfg)

        self.vpred_layer = nn.Sequential(
            nn.Linear(self.d, int(self.d // 2), bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(int(self.d // 2), self.out_dim, bias=True))

    def forward(self, o_g_emb):
        qpred = self.vpred_layer(o_g_emb)
        return qpred


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class SFSPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, env, cfg):
        super().__init__()
        self.device = device
        self.batch_size = cfg.policy.batch_size
        self.ac_dim = len(ATOM_IDX) + 1
        self.emb_size = cfg.policy.emb_dim
        self.tau = cfg.policy.tau

        # init candidate atoms
        self.bond_type_num = 3

        self.env = env  # env utilized to init cand motif mols
        # self.cand = self.create_candidate_motifs()
        # self.cand_g = dgl.batch([x['g'] for x in self.cand])
        # self.cand_ob_len = self.cand_g.batch_num_nodes().tolist()
        # Create candidate descriptors

        # if args.desc == 'ecfp':
        #     desc = ecfp
        #     self.desc_dim = 1024
        # elif args.desc == 'desc':
        #     desc = rdkit_descriptors
        #     self.desc_dim = 199
        # self.cand_desc = torch.Tensor([desc(Chem.MolFromSmiles(x['smi'])) for x in self.cand]).to(self.device)
        # self.motif_type_num = len(self.cand)

        self.action_layer_type_emb = nn.Embedding(num_embeddings=len(ATOM_IDX)+1, embedding_dim=cfg.policy.atom_emb_dim)

        self.action_layers_type = nn.Sequential(
            nn.Linear(cfg.emb.emb_dim, cfg.policy.emb_dim // 2, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.policy.emb_dim // 2, len(ATOM_IDX)+1, bias=True)).to(self.device)

        self.action_layers_atom = nn.ModuleList(
            [nn.Bilinear(cfg.policy.atom_emb_dim, cfg.policy.emb_dim, cfg.policy.emb_dim).to(self.device),
             nn.Linear(cfg.policy.atom_emb_dim, cfg.policy.emb_dim, bias=False).to(self.device),
             nn.Linear(cfg.policy.emb_dim, cfg.policy.emb_dim, bias=False).to(self.device),
             nn.Sequential(nn.Linear(cfg.policy.emb_dim, cfg.policy.emb_dim // 2, bias=False),
                           nn.ReLU(inplace=False),
                           nn.Linear(cfg.policy.emb_dim // 2, 1, bias=True)).to(self.device)])

        self.action_layers_bond = nn.ModuleList([
            nn.Bilinear(cfg.policy.atom_emb_dim, cfg.policy.emb_dim, cfg.policy.emb_dim).to(self.device),
            nn.Linear(cfg.policy.atom_emb_dim, cfg.policy.emb_dim, bias=False).to(self.device),
            nn.Linear(cfg.policy.emb_dim, cfg.policy.emb_dim, bias=False).to(self.device),
            nn.MultiheadAttention(embed_dim=cfg.policy.emb_dim, num_heads=4, batch_first=True).to(self.device),
            nn.Sequential(nn.Linear(cfg.policy.emb_dim, cfg.policy.emb_dim // 2, bias=False),
                          nn.ReLU(inplace=False),
                          nn.Linear(cfg.policy.emb_dim // 2, len(BOND_IDX)+1, bias=True)).to(self.device)
        ])

        # self.action1_layers = nn.ModuleList(
        #     [nn.Bilinear(2 * args.emb_size, 2 * args.emb_size, args.emb_size).to(self.device),
        #      nn.Linear(2 * args.emb_size, args.emb_size, bias=False).to(self.device),
        #      nn.Linear(2 * args.emb_size, args.emb_size, bias=False).to(self.device),
        #      nn.Sequential(nn.Linear(args.emb_size, args.emb_size // 2, bias=False),
        #                    nn.ReLU(inplace=False),
        #                    nn.Linear(args.emb_size // 2, 1, bias=True)).to(self.device)])
        #
        # self.action2_layers = nn.ModuleList([nn.Bilinear(self.desc_dim, args.emb_size, args.emb_size).to(self.device),
        #                                      nn.Linear(self.desc_dim, args.emb_size, bias=False).to(self.device),
        #                                      nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device),
        #                                      nn.Sequential(nn.Linear(args.emb_size, args.emb_size, bias=False),
        #                                                    nn.ReLU(inplace=False),
        #                                                    nn.Linear(args.emb_size, args.emb_size, bias=True),
        #                                                    nn.ReLU(inplace=False),
        #                                                    nn.Linear(args.emb_size, 1, bias=True), )
        #                                      ])
        #
        # self.action3_layers = nn.ModuleList(
        #     [nn.Bilinear(2 * args.emb_size, 2 * args.emb_size, args.emb_size).to(self.device),
        #      nn.Linear(2 * args.emb_size, args.emb_size, bias=False).to(self.device),
        #      nn.Linear(2 * args.emb_size, args.emb_size, bias=False).to(self.device),
        #      nn.Sequential(
        #          nn.Linear(args.emb_size, args.emb_size // 2, bias=False),
        #          nn.ReLU(inplace=False),
        #          nn.Linear(args.emb_size // 2, 1, bias=True)).to(self.device)])

        # Zero padding with max number of actions
        self.max_action = 40  # max atoms

    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10,
                       dim: int = -1, g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, deterministic=False):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """

        g.ndata['node_emb'] = node_emb
        batch_num_nodes_list = g.batch_num_nodes().tolist()

        # ===============================
        # step 1 : where to add
        # ===============================
        # select only nodes with attachment points

        logit_first = self.action_layers_type(graph_emb)
        ac_first_prob = F.softmax(logit_first, dim=1) + 1e-8
        log_ac_first_prob = ac_first_prob.log()
        ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, dim=0).transpose(0, 1)
        ac_first = torch.argmax(ac_first_hot, dim=-1)

        # ===============================
        # step 2 : which motif to add - Using Descriptors
        # ===============================

        cand_action = torch.tensor([i for i in range(len(ATOM_IDX)+1)], dtype=torch.long, device=device)
        action_type_emb = self.action_layer_type_emb(cand_action)
        emb_first = torch.matmul(ac_first_hot.t(), action_type_emb)
        emb_first_expand = []
        for i, num_nodes in enumerate(batch_num_nodes_list):
            emb_first_expand.append(emb_first[i].repeat(num_nodes, 1))
        emb_first_expand = torch.cat(emb_first_expand, dim=0)
        emb_cat = self.action_layers_atom[0](emb_first_expand, node_emb) + \
                  self.action_layers_atom[1](emb_first_expand) + self.action_layers_atom[2](node_emb)
        logit_second = self.action_layers_atom[3](emb_cat)

        att_mask = g.ndata['att_mask']  # used to select att embs from node embs
        logit_second = torch.mul(logit_second.view(-1), att_mask.to(torch.float)).view(-1, 1)

        ac_second_prob = [torch.softmax(logit, dim=0)
                          for i, logit in enumerate(torch.split(logit_second, g.batch_num_nodes(), dim=0))]
        ac_second_prob = [p + 1e-8 for p in ac_second_prob]
        log_ac_second_prob = [x.log() for x in ac_second_prob]

        ac_second_hot = [self.gumbel_softmax(ac_second_prob[i], tau=self.tau, hard=True, dim=0)
                         for i in range(g.batch_size)]
        ac_second = []
        for i, node_emb_part in enumerate(torch.split(node_emb, g.batch_num_nodes(), dim=0)):
            ac_second.append(torch.argmax(ac_second_hot[i], dim=-1))
        ac_second_prob = [torch.cat([ac_second_prob[i], ac_second_prob[i].
                                    new_zeros(max(MAX_NODE - ac_second_prob[i].size(0), 0), 1)], 0).
                              contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        log_ac_second_prob = [torch.cat([log_ac_second_prob[i], log_ac_second_prob[i].
                                    new_zeros(max(MAX_NODE - log_ac_second_prob[i].size(0), 0), 1)], 0).
                              contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        print(ac_second)
        print(ac_second_prob)
        ac_second = torch.cat(ac_second)
        ac_second_prob = torch.cat(ac_second_prob)
        log_ac_second_prob = torch.cat(log_ac_second_prob)

        # ===============================
        # step 4 : where to add on motif
        # ===============================
        # Select att points from candidate
        emb_cat = self.action_layers_bond[0](emb_first_expand, node_emb) + \
                  self.action_layers_bond[1](emb_first_expand) + self.action_layers_bond[2](node_emb)
        emb_cat_ext = []
        for i, node_emb_part in enumerate(torch.split(emb_cat, g.batch_num_nodes(), dim=0)):
            emb_cat_ext.append(torch.cat([node_emb_part, node_emb_part.new_zeros(
                max(MAX_NODE - node_emb_part.size(0), 0), node_emb_part.size(1))], 0).contiguous().view(1, MAX_NODE, node_emb_part.size(1)))
        emb_cat_ext = torch.cat(emb_cat_ext)
        h_ac_third, _ = self.action_layers_bond[3](emb_cat_ext, emb_cat_ext, emb_cat_ext)
        logit_third = self.action_layers_bond[4](h_ac_third)
        logit_third_ext = []
        ac_third_prob = []
        log_ac_third_prob = []
        ac_third_hot = []
        ac_third = []
        for i in range(g.batch_size):
            logit_third_ext.append(logit_third[i, :batch_num_nodes_list[i]])
            ac_third_prob.append(F.softmax(logit_third[i, :batch_num_nodes_list[i]], dim=1)+1e-8)
            log_ac_third_prob.append(ac_third_prob[-1].log())
            ac_third_hot.append(self.gumbel_softmax(ac_third_prob[-1], tau=self.tau, hard=True, dim=1))
            ac_third.append(torch.argmax(ac_third_hot[-1], dim=1))
        ac_third_prob = [torch.cat([ac_third_prob[i], ac_third_prob[i].
                                    new_zeros(max(MAX_NODE - ac_third_prob[i].size(0), 0), ac_third_prob[i].size(1))], 0).
                              contiguous().view(1, MAX_NODE, ac_third_prob[i].size(1)) for i in range(g.batch_size)]
        log_ac_third_prob = [torch.cat([log_ac_third_prob[i], log_ac_third_prob[i].
                                        new_zeros(max(MAX_NODE - log_ac_third_prob[i].size(0), 0), log_ac_third_prob[i].size(1))], 0).
                                  contiguous().view(1, MAX_NODE, log_ac_third_prob[i].size(1)) for i in range(g.batch_size)]
        ac_third_prob = torch.cat(ac_third_prob)
        log_ac_third_prob = torch.cat(log_ac_third_prob)

        # ==== concat everything ====
        ac_prob = {"first": ac_first_prob, "second": ac_second_prob, "third": ac_third_prob}
        log_ac_prob = {"first": log_ac_first_prob, "second": log_ac_second_prob, "third": log_ac_third_prob}
        ac = {"first": ac_first, "second": ac_second, "third": ac_third}

        # ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        # log_ac_prob = torch.cat([log_ac_first_prob,
        #                          log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        # ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, ac_prob, log_ac_prob

    def _distribution(self, ac_prob):
        dists = []
        dists.append(Categorical(probs=ac_prob["first"]))
        dists.append(Categorical(probs=ac_prob["second"]))
        for i in range(ac_prob["third"].shape[0]):
            dists.append(Categorical(probs=ac_prob["third"][i]))

        # ac_prob_split = torch.split(ac_prob, [len(ATOM_IDX)+1, MAX_NODE, MAX_NODE], dim=1)
        # dists = [Categorical(probs=pr) for pr in ac_prob_split]
        return dists

    def _log_prob_from_distribution(self, dists, act):
        log_probs = [p.log_prob(act[0][i]) for i, p in enumerate(dists)]

        return torch.cat(log_probs, dim=0)

    def sample(self, ac, graph_emb, node_emb, g):
        g.ndata['node_emb'] = node_emb
        batch_num_nodes_list = g.batch_num_nodes().tolist()

        # ===============================
        # step 1 : where to add
        # ===============================
        # select only nodes with attachment points

        logit_first = self.action_layers_type(graph_emb)
        ac_first_prob = F.softmax(logit_first, dim=1) + 1e-8
        log_ac_first_prob = ac_first_prob.log()

        # ===============================
        # step 2 : which motif to add - Using Descriptors
        # ===============================

        cand_action = torch.tensor([i for i in range(len(ATOM_IDX) + 1)], dtype=torch.long)
        action_type_emb = self.action_layer_type_emb(cand_action)
        emb_first = torch.matmul(ac["first"], action_type_emb)
        emb_first_expand = []
        for i, num_nodes in enumerate(range(batch_num_nodes_list)):
            emb_first_expand.append(emb_first[i].repeat(num_nodes, 1))
        emb_first_expand = torch.cat(emb_first_expand, dim=0)
        emb_cat = self.action_layers_atom[0](emb_first_expand, node_emb) + \
                  self.action_layers_atom[1](emb_first_expand) + self.action_layers_atom[2](node_emb)
        logit_second = self.action_layers_atom[3](emb_cat)

        att_mask = g.ndata['att_mask']  # used to select att embs from node embs
        ac_second_prob = [torch.softmax(logit, dim=0)
                          for i, logit in enumerate(torch.split(logit_second, g.batch_num_nodes(), dim=0))]
        ac_second_prob = [p + 1e-8 for p in ac_second_prob]
        log_ac_second_prob = [x.log() for x in ac_second_prob]

        ac_second_prob = [torch.cat([ac_second_prob[i], ac_second_prob[i].
                                    new_zeros(max(MAX_NODE - ac_second_prob[i].size(0), 0), 1)], 0).
                              contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        log_ac_second_prob = [torch.cat([log_ac_second_prob[i], log_ac_second_prob[i].
                                        new_zeros(max(MAX_NODE - log_ac_second_prob[i].size(0), 0), 1)], 0).
                                  contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        ac_second_prob = torch.cat(ac_second_prob)
        log_ac_second_prob = torch.cat(log_ac_second_prob)

        # ===============================
        # step 4 : where to add on motif
        # ===============================
        # Select att points from candidate
        emb_cat = self.action_layers_bond[0](emb_first_expand, node_emb) + \
                  self.action_layers_bond[1](emb_first_expand) + self.action_layers_bond[2](node_emb)
        emb_cat_ext = []
        for i, node_emb_part in enumerate(torch.split(emb_cat, g.batch_num_nodes(), dim=0)):
            emb_cat_ext.append(torch.cat([node_emb_part, node_emb_part.new_zeros(
                max(MAX_NODE - node_emb_part.size(0), 0), 1)], 0).contiguous().view(1, MAX_NODE))
        emb_cat_ext = torch.cat(emb_cat_ext)
        h_ac_third, _ = self.action_layers_bond[3](emb_cat_ext)
        logit_third = self.action_layers_bond[4](h_ac_third)
        logit_third_ext = []
        ac_third_prob = []
        log_ac_third_prob = []
        for i in range(g.batch_size):
            logit_third_ext.append(logit_third[i, :batch_num_nodes_list[i]])
            ac_third_prob.append(F.softmax(logit_third[i, :batch_num_nodes_list[i]], dim=1) + 1e-8)
            log_ac_third_prob.append(ac_third_prob[-1].log())
        ac_third_prob = [torch.cat([ac_third_prob[i], ac_third_prob[i].
                                   new_zeros(max(MAX_NODE - ac_third_prob[i].size(0), 0), 1)], 0).
                             contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        log_ac_third_prob = [torch.cat([log_ac_third_prob[i], log_ac_third_prob[i].
                                       new_zeros(max(MAX_NODE - log_ac_third_prob[i].size(0), 0), 1)], 0).
                                 contiguous().view(1, MAX_NODE) for i in range(g.batch_size)]
        ac_third_prob = torch.cat(ac_third_prob)
        log_ac_third_prob = torch.cat(log_ac_third_prob)

        # ==== concat everything ====
        ac_prob = {"first": ac_first_prob, "second": ac_second_prob, "third": ac_third_prob}
        log_ac_prob = {"first": log_ac_first_prob, "second": log_ac_second_prob, "third": log_ac_third_prob}

        return ac_prob, log_ac_prob
