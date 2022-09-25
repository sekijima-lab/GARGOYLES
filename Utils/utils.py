import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import pickle
import networkx as nx
import argparse
from statistics import mean
from scipy.sparse.csgraph import connected_components
import itertools
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import warnings

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, QED, DataStructs, BRICS, Descriptors, Crippen
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

from Utils.mol_utils import Mol2Mat

RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

atoms = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
         "[Sc]", "Ti", "V", "Cr", "[Mn]", "Fe", "[Co]", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
         "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "[Sn]", "Sb", "Te", "I", "Xe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def atom_removal_mask(mol):
    mask = [1] * mol.GetNumAtoms()
    for i in range(len(mask)):
        rwmol = RWMol.RWMol(mol)
        for atom in rwmol.GetAtoms():
            map_id = atom.GetAtomMapNum()
            if map_id == i + 1:
                rwmol.RemoveAtom(atom.GetIdx())
        out_mol = rwmol.GetMol()
        smiles = Chem.MolToSmiles(out_mol)

        if "." in smiles:
            mask[i] = 0

    return mask


def setBFSorder(mol, start_id=0):
    index = {}
    bfs_queue = [start_id]
    visited_node = [start_id]
    ind_counter = 1

    while len(bfs_queue) > 0:
        c_node = bfs_queue[0]
        index[c_node] = ind_counter
        ind_counter += 1
        bfs_queue = bfs_queue[1:]

        for atom in mol.GetAtomWithIdx(c_node).GetNeighbors():
            id = atom.GetIdx()
            if id not in visited_node:
                bfs_queue.append(id)
                visited_node.append(id)

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(index[i])

    return mol


def Mat2DGLGraph(adj, node_feat):
    G = nx.from_numpy_matrix(adj)
    G = G.to_directed()
    g = dgl.from_networkx(G, edge_attrs=['weight'])
    g.ndata["atom"] = torch.tensor(node_feat, dtype=torch.long)

    return g


def clear_atommap(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def checkRadicals(mol):
    for atom in mol.GetAtoms():
        print(atom.GetNumRadicalElectrons(), atom.GetFormalCharge(), atom.GetExplicitValence(),
              atom.GetImplicitValence(), atom.GetTotalNumHs(), atom.GetSymbol(), atom.GetIsAromatic(),
              atom.GetAtomMapNum())
        if atom.GetNumRadicalElectrons() == 1 and atom.GetFormalCharge() == 1:
            atom.SetNumRadicalElectrons(0)
            atom.SetFormalCharge(0)

        # if atom.GetExplicitValence() + atom.Get


def neutralizeRadicals(mol):
    for a in mol.GetAtoms():
        print(a.GetNumRadicalElectrons(), a.GetFormalCharge())
        if a.GetNumRadicalElectrons() == 1 and a.GetFormalCharge() == 1:
            a.SetNumRadicalElectrons(0)
            a.SetFormalCharge(0)


def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list


def DrawMol(mol, out_path, out_filename="sample.png", size=(1000, 1000)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    option = drawer.drawOptions()
    option.addAtomIndices = True

    drawer.DrawMolecule(tm)
    drawer.FinishDrawing()

    img = drawer.GetDrawingText()
    with open(out_path + out_filename, mode="wb") as f:
        f.write(img)


def convert_smiles(smiles, vocab, mode):
    """
    :param smiles:
    :param vocab: dict of tokens
    :param mode: s2i: string -> int
                 i2s: int -> string
    :return: converted smiles,
    """
    converted = []
    if mode == "s2i":
        for token in smiles:
            converted.append(vocab.index(token))
    elif mode == "i2s":
        for ind in smiles:
            converted.append(vocab[ind])
    return converted


def parse_smiles(smiles):
    parsed = []
    i = 0
    while i < len(smiles):
        asc = ord(smiles[i])
        if 64 < asc < 91:
            if i != len(smiles) - 1 and smiles[i:i + 2] in atoms:
                parsed.append(smiles[i:i + 2])
                i += 2
            else:
                parsed.append(smiles[i])
                i += 1
        elif asc == 91:
            j = i
            while smiles[i] != "]":
                i += 1
            i += 1
            parsed.append(smiles[j:i])

        else:
            parsed.append(smiles[i])
            i += 1

    return parsed


def main():
    smiles = "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1"
    mol = Chem.MolFromSmiles(smiles)
    mol = setBFSorder(mol)
    adj, node_feat = Mol2Mat(mol)
    g = Mat2DGLGraph(adj, node_feat)
    print(g)
    print(g.ndata["atom"])
    print(g.edata["weight"])


if __name__ == '__main__':
    main()
