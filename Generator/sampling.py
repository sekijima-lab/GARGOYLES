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

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, QED, DataStructs, BRICS, Descriptors, Crippen
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
from dgl.dataloading import GraphDataLoader

from Utils.utils import device
from Utils.mol_utils import ATOM_IDX, Mol2Graph, setBFSorder, MAX_NODE, BOND_IDX, clear_atommap, ATOM_IDX_INV, \
    VALENCY, RDKIT_BOND_INV
from Model.model import FeatureExtractor, NodeHeader, EdgeRNNHeader
from Model.train import reverseNodeIdx, reshapeRNNInput


def dice_similarity_coefficient(list_a, list_b):
    set_intersection = set.intersection(set(list_a), set(list_b))
    num_intersection = len(set_intersection)

    num_listA = len(list_a)
    num_listB = len(list_b)

    try:
        return float(2.0 * num_intersection) / (num_listA + num_listB)
    except ZeroDivisionError:
        return 1.0


class Sampler:
    def __init__(self, cfg, model_dir, model_ver):
        super(Sampler, self).__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(node_in_feats=cfg["emb"]["infeat_atom"],
                                                  edge_in_feats=cfg["emb"]["infeat_bond"],
                                                  hidden_feats=cfg["emb"]["hidden"]).to(device)
        self.node_header = NodeHeader(in_channels=cfg["node"]["in_channels"],
                                      hidden_channels=cfg["node"]["hidden_channels"],
                                      out_channels=cfg["node"]["out_channels"]).to(device)
        self.edge_header = EdgeRNNHeader(num_embeddings=cfg["edge"]["num_embeddings"],
                                         embedding_dim=cfg["edge"]["embedding_dim"],
                                         h_graph_dim=cfg["edge"]["h_graph_dim"],
                                         input_size_rnn=cfg["edge"]["input_size_rnn"],
                                         hidden_size_rnn=cfg["edge"]["hidden_size_rnn"],
                                         num_layers=cfg["edge"]["num_layers"],
                                         hidden_size_head=cfg["edge"]["hidden_size_head"],
                                         out_size_head=cfg["edge"]["out_size_head"], seq_len=MAX_NODE).to(device)
        self.generaed_list = []

        # Model Loading
        self.feature_extractor.load_state_dict(torch.load(hydra.utils.get_original_cwd() + model_dir
                                                          + f"feature-extractor-ep{model_ver}.pth"))
        self.node_header.load_state_dict(torch.load(hydra.utils.get_original_cwd() + model_dir
                                                    + f"node-header-ep{model_ver}.pth"))
        self.edge_header.load_state_dict(torch.load(hydra.utils.get_original_cwd() + model_dir
                                                    + f"edge-header-ep{model_ver}.pth"))

    def sample(self, smiles, n):
        generaed_list = []
        for i in range(n):
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            batch_num_node = [mol.GetNumAtoms()]
            while mol.GetNumAtoms() < MAX_NODE:
                mol = setBFSorder(mol)
                try:
                    g = Mol2Graph(mol)
                except RuntimeError:
                    break
                bg = dgl.batch([g]).to(device)

                # Feature extraction
                h_graph, h_node = self.feature_extractor(bg, bg.ndata["atomic"], bg.edata["type"])

                # Node type prediction
                y_node = self.node_header(h_graph)
                prob_node = F.softmax(y_node, dim=1).cpu().detach().numpy()
                pred_atom = np.random.choice([j for j in range(len(ATOM_IDX)+1)], p=prob_node[0])
                batch_label_node = torch.tensor([pred_atom], dtype=torch.long)

                if pred_atom == len(ATOM_IDX):
                    break

                input_rnn = reverseNodeIdx(h_node, bg.batch_num_nodes().tolist())
                input_rnn = reshapeRNNInput(input_rnn, bg.batch_num_nodes().tolist())
                y_edge = self.edge_header(input_rnn.to(device), batch_num_node, h_graph, batch_label_node.view(-1).to(device))
                prob_edge = F.softmax(y_edge, dim=2).cpu().detach().numpy()[0]
                pred_edge = []
                # print(prob_edge)
                for j in range(prob_edge.shape[0]):
                    pred_btype = np.random.choice([k for k in range(len(BOND_IDX)+1)], p=prob_edge[j])
                    pred_edge.append(pred_btype)

                if sum(pred_edge) == 0:
                    break

                next_mol = self.revise_mol(mol, [pred_atom], [pred_edge])
                if next_mol is not None:
                    mol = next_mol[0]
                else:
                    break

                batch_num_node[0] += 1

            mol = clear_atommap(mol)
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                pass
            generaed_list.append(Chem.MolToSmiles(mol))

        return generaed_list

    def step(self, mol, n=1):
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol = setBFSorder(mol)
        batch_num_node = [mol.GetNumAtoms()]*n
        g = Mol2Graph(mol)
        bg = dgl.batch([g for _ in range(n)]).to(device)

        # Feature extraction
        h_graph, h_node = self.feature_extractor(bg, bg.ndata["atomic"], bg.edata["type"])

        # Node type prediction
        pred_atom_list = []
        y_node = self.node_header(h_graph)
        prob_node = F.softmax(y_node, dim=1).cpu().detach().numpy()
        for _ in range(n):
            pred_atom = np.random.choice([j for j in range(len(ATOM_IDX) + 1)], p=prob_node[0])
            pred_atom_list.append(pred_atom)

        pred_edge_list = []
        batch_label_node = torch.tensor(pred_atom_list, dtype=torch.long)
        input_rnn = reverseNodeIdx(h_node, bg.batch_num_nodes().tolist())
        input_rnn = reshapeRNNInput(input_rnn, bg.batch_num_nodes().tolist())
        y_edge = self.edge_header(input_rnn.to(device), batch_num_node, h_graph, batch_label_node.view(-1).to(device))
        prob_edge = F.softmax(y_edge, dim=2).cpu().detach().numpy()
        for i in range(n):
            pred_edge = []
            # print(prob_edge)
            for j in range(prob_edge.shape[1]):
                pred_btype = np.random.choice([k for k in range(len(BOND_IDX) + 1)], p=prob_edge[i, j])
                pred_edge.append(pred_btype)
            pred_edge_list.append(pred_edge)

        return pred_atom_list, pred_edge_list

    def revise_mol(self, mol, atom_type_list, edges_type_list):
        # print(Chem.MolToSmiles(mol))
        # print(atom_type_list)
        # print(edges_type_list)
        mol_list = []
        for j in range(len(atom_type_list)):
            rwmol = RWMol(mol)

            # Add atom
            new_atom_id = rwmol.AddAtom(Chem.Atom(ATOM_IDX_INV[atom_type_list[j]]))

            # Add bonds
            new_atom_valence = VALENCY[ATOM_IDX_INV[atom_type_list[j]]]
            for i in range(mol.GetNumAtoms()):
                existing_atom = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() == mol.GetNumAtoms()-i][0]
                existing_atom_valence = existing_atom.GetExplicitValence()
                if 0 < edges_type_list[j][i] <= min(new_atom_valence, VALENCY[existing_atom.GetSymbol()] - existing_atom_valence):
                    rwmol.AddBond(new_atom_id, existing_atom.GetIdx(), RDKIT_BOND_INV[edges_type_list[j][i]])
                    new_atom_valence -= edges_type_list[j][i]
            if new_atom_valence == VALENCY[ATOM_IDX_INV[atom_type_list[j]]]:
                return None
            rwmol.UpdatePropertyCache()
            mol_list.append(rwmol.GetMol())

        return mol_list


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    sampler = Sampler(cfg, model_dir=cfg["sample"]["model_dir"], model_ver=cfg["sample"]["model_ver"])
    generaed_list = sampler.sample(smiles=cfg["sample"]["seed_smiles"], n=cfg["sample"]["num"])
    generaed_list = list(set(generaed_list))

    for smiles in generaed_list:
        print(smiles)


if __name__ == '__main__':
    main()

