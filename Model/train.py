import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from config.config import cs

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.dataloading import GraphDataLoader

from Utils.utils import device
from Utils.mol_utils import VOCABULARY, ATOM_IDX, MAX_NODE, BOND_IDX
from Model.dataset import FragmentDataset, collate_fragment
from Model.model import FeatureExtractor, NodeHeader, EdgeRNNHeader


def reverseNodeIdx(x, batch_num_nodes):
    id_list = []
    for i in range(len(batch_num_nodes)):
        offset = sum(batch_num_nodes[:i])
        num_nodes = batch_num_nodes[i]
        id_list.extend([num_nodes-1-j+offset for j in range(num_nodes)])

    return x[id_list, ]


def reshapeRNNInput(x, batch_num_nodes):
    in_rnn_tensor = torch.zeros([len(batch_num_nodes), MAX_NODE, x.shape[1]])
    for i in range(len(batch_num_nodes)):
        offset = sum(batch_num_nodes[:i])
        in_rnn_tensor[i, :batch_num_nodes[i]] = x[offset: batch_num_nodes[i]+offset]

    return in_rnn_tensor


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc)

    return list_topk_accs


def train(cfg: DictConfig) -> None:
    # Dataset setting
    path = "/data/features/freed/"
    with open(hydra.utils.get_original_cwd() + path+"dgl-graph.pickle", mode="rb") as f:
        graph_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + path+"label.pickle", mode="rb") as f:
        label_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + path+"num-node.pickle", mode="rb") as f:
        num_node_list = pickle.load(f)

    sp = int(len(graph_list)*1)
    train_set = FragmentDataset(graph_list[:sp], label_list[:sp], num_node_list[:sp], name="Train")
    test_set = FragmentDataset(graph_list[sp:], label_list[sp:], num_node_list[sp:], name="Test")
    test_set = train_set
    train_loader = GraphDataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                                   shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                   collate_fn=collate_fragment)
    test_loader = GraphDataLoader(test_set, batch_size=cfg["train"]["batch_size"],
                                  shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                  collate_fn=collate_fragment)

    # Model setting
    feature_extractor = FeatureExtractor(node_in_feats=cfg["emb"]["infeat_atom"],
                                         edge_in_feats=cfg["emb"]["infeat_bond"],
                                         hidden_feats=cfg["emb"]["hidden"]).to(device)
    node_header = NodeHeader(in_channels=cfg["node"]["in_channels"], hidden_channels=cfg["node"]["hidden_channels"],
                             out_channels=cfg["node"]["out_channels"]).to(device)
    edge_header = EdgeRNNHeader(num_embeddings=cfg["edge"]["num_embeddings"], embedding_dim=cfg["edge"]["embedding_dim"],
                                h_graph_dim=cfg["edge"]["h_graph_dim"], input_size_rnn=cfg["edge"]["input_size_rnn"],
                                hidden_size_rnn=cfg["edge"]["hidden_size_rnn"], num_layers=cfg["edge"]["num_layers"],
                                hidden_size_head=cfg["edge"]["hidden_size_head"], out_size_head=cfg["edge"]["out_size_head"],
                                seq_len=MAX_NODE).to(device)

    optimizer = optim.Adam(list(feature_extractor.parameters())
                           +list(node_header.parameters())+list(edge_header.parameters()), lr=cfg["train"]["lr"])
    criterion_atom = nn.CrossEntropyLoss()
    criterion_bond = nn.CrossEntropyLoss(ignore_index=len(BOND_IDX)+1)

    # Log
    mlflow.set_tracking_uri("file:/" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.start_run()
    mlflow.log_param("batch_size", cfg["train"]["batch_size"])
    mlflow.log_param("lr", cfg["train"]["lr"])
    mlflow.log_param("epoch", cfg["train"]["epoch"])
    mlflow.log_param("data_size", cfg["train"]["data_size"])

    num_epochs = cfg["train"]["start_epoch"]+cfg["train"]["epoch"]
    for step in range(1+cfg["train"]["start_epoch"], num_epochs+1):
        # Training
        feature_extractor.train()
        node_header.train()
        edge_header.train()
        train_loss = []
        train_loss_node = []
        train_loss_edge = []
        train_acc_node = []
        train_acc_edge = []

        with tqdm(total=train_loader.__len__(), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{step}/{num_epochs}](Train)")

            for t, (batch_graph, batch_label_node, batch_label_edge, batch_num_node) in enumerate(train_loader):
                batch_graph = batch_graph.to(device)
                h_graph, h_node = feature_extractor(batch_graph, batch_graph.ndata["atomic"], batch_graph.edata["type"])

                y_node = node_header(h_graph)
                input_rnn = reverseNodeIdx(h_node, batch_graph.batch_num_nodes().tolist())
                input_rnn = reshapeRNNInput(input_rnn, batch_graph.batch_num_nodes().tolist())
                y_edge = edge_header(input_rnn.to(device), batch_num_node, h_graph, batch_label_node.view(-1).to(device))

                loss_node = criterion_atom(y_node, batch_label_node.view(-1).to(device))
                loss_edge = criterion_bond(y_edge.reshape(-1, y_edge.shape[2]), batch_label_edge.view(-1).to(device))
                loss = loss_node + loss_edge

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(batch_label_edge.view(-1).cpu().numpy())
                acc_node = accuracy(F.softmax(y_node.reshape(-1, len(ATOM_IDX)+1), dim=1),
                                    batch_label_node.view(-1).to(device), topk=(1, ))[0]
                acc_edge = accuracy(F.softmax(y_edge.reshape(-1, len(BOND_IDX) + 1), dim=1),
                                    batch_label_edge.view(-1).to(device), topk=(1,))[0]

                train_loss.append(float(loss))
                train_loss_node.append(float(loss_node))
                train_loss_edge.append(float(loss_edge))
                train_acc_node.append(acc_node.cpu().item())
                train_acc_edge.append(acc_edge.cpu().item())
                pbar.set_postfix({"total loss": train_loss[-1], "atom loss": train_loss_node[-1],
                                  "bond loss": train_loss_edge[-1], "atom-accuracy": train_acc_node[-1],
                                  "bond-accuracy": train_acc_edge[-1]})
                pbar.update(1)

        # Evaluation
        feature_extractor.eval()
        node_header.eval()
        edge_header.eval()
        test_loss = []
        test_loss_node = []
        test_loss_edge = []
        test_acc_node = []
        test_acc_edge = []

        with torch.no_grad():
            with tqdm(total=test_loader.__len__(), unit="batch") as pbar:
                pbar.set_description(f"Epoch[{step}/{num_epochs}](Test)")

                for t, (batch_graph, batch_label_node, batch_label_edge, batch_num_node) in enumerate(test_loader):
                    batch_graph = batch_graph.to(device)
                    h_graph, h_node = feature_extractor(batch_graph, batch_graph.ndata["atomic"], batch_graph.edata["type"])

                    y_node = node_header(h_graph)
                    input_rnn = reverseNodeIdx(h_node, batch_graph.batch_num_nodes().tolist())
                    input_rnn = reshapeRNNInput(input_rnn, batch_graph.batch_num_nodes().tolist())
                    y_edge = edge_header(input_rnn.to(device), batch_num_node, h_graph,
                                         batch_label_node.view(-1).to(device))

                    loss_node = criterion_atom(y_node, batch_label_node.view(-1).to(device))
                    loss_edge = criterion_bond(y_edge.reshape(-1, y_edge.shape[2]), batch_label_edge.view(-1).to(device))
                    loss = loss_node + loss_edge

                    acc_node = accuracy(F.softmax(y_node.reshape(-1, len(ATOM_IDX) + 1), dim=1),
                                        batch_label_node.view(-1).to(device), topk=(1,))[0]
                    acc_edge = accuracy(F.softmax(y_edge.reshape(-1, len(BOND_IDX) + 1), dim=1),
                                        batch_label_edge.view(-1).to(device), topk=(1,))[0]

                    test_loss.append(float(loss))
                    test_loss_node.append(float(loss_node))
                    test_loss_edge.append(float(loss_edge))
                    test_acc_node.append(acc_node.cpu().item())
                    test_acc_edge.append(acc_edge.cpu().item())
                    pbar.set_postfix({"total loss": test_loss[-1], "atom loss": test_loss_node[-1],
                                      "bond loss": test_loss_edge[-1], "atom-accuracy": test_acc_node[-1],
                                      "bond-accuracy": test_acc_edge[-1]})
                    pbar.update(1)

        if step % cfg["train"]["save_step"] == 0:
            torch.save(feature_extractor.state_dict(),
                       hydra.utils.get_original_cwd()+cfg["train"]["model_dir"]+f"feature-extractor-ep{step}.pth")
            torch.save(node_header.state_dict(),
                       hydra.utils.get_original_cwd()+cfg["train"]["model_dir"] + f"node-header-ep{step}.pth")
            torch.save(edge_header.state_dict(),
                       hydra.utils.get_original_cwd() + cfg["train"]["model_dir"] + f"edge-header-ep{step}.pth")

        # Log
        mlflow.log_metric("Train Total loss", float(np.mean(train_loss)), step=step)
        mlflow.log_metric("Train Atom loss", float(np.mean(train_loss_node)), step=step)
        mlflow.log_metric("Train Bond loss", float(np.mean(train_loss_edge)), step=step)
        mlflow.log_metric("Train Atom accuracy", float(np.mean(train_acc_node)), step=step)
        mlflow.log_metric("Train Bond accuracy", float(np.mean(train_acc_edge)), step=step)
        mlflow.log_metric("Test loss", float(np.mean(test_loss)), step=step)
        mlflow.log_metric("Test Atom loss", float(np.mean(test_loss_node)), step=step)
        mlflow.log_metric("Test Bond loss", float(np.mean(test_loss_edge)), step=step)
        mlflow.log_metric("Test Atom accuracy", float(np.mean(test_acc_node)), step=step)
        mlflow.log_metric("Test Bond accuracy", float(np.mean(test_acc_edge)), step=step)

    with open(hydra.utils.get_original_cwd()+cfg["train"]["log_dir"]+cfg["train"]["log_filename"], "w") as f:
        f.write(cfg["train"]["log_filename"])
    mlflow.log_artifact(hydra.utils.get_original_cwd()+cfg["train"]["log_dir"]+cfg["train"]["log_filename"])
    mlflow.end_run()


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()


