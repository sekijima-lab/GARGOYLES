import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import read_smilesset
from Utils.mol_utils import valence_mask, add_frag, ATOM_IDX, RDKIT_BOND, RDKIT_BOND_INV, Mol2Graph, \
    setBFSorder, clear_atommap, MAX_NODE, BOND_IDX, parse_ring
from Utils.reward import getReward

import collections
from tqdm import tqdm
import pickle
import time
from copy import deepcopy

import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from rdkit.Chem.rdchem import RWMol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


out_path = "../data/features/pretrain/"


def parse_brics(mol):
    frag_mol = BRICS.BreakBRICSBonds(mol)
    frag = [x for x in Chem.MolToSmiles(frag_mol).split(".")]

    frag_maps = []
    frag_mask = [0]*len(frag)
    for i, s in enumerate(frag):
        mol = Chem.MolFromSmiles(s)
        maps = []
        for atom in mol.GetAtoms():
            mid = atom.GetAtomMapNum()
            if mid > 0:
                maps.append(atom.GetAtomMapNum())
        frag_maps.append(maps)

        if s.count("*") == 1:
            frag_mask[i] = 1

    return frag_maps, frag_mask


def make_vocabulary(out_path="data/features/", parse_type="brics"):
    smiles_list = read_smilesset("data/zinc_250k.smi")
    vocab_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
        mol = setBFSorder(mol)

        if parse_type == "brics":
            frag = [x for x in Chem.MolToSmiles(BRICS.BreakBRICSBonds(mol)).split(".")]

            for frag_smiles in frag:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                rm_ids = []
                for atom in frag_mol.GetAtoms():
                    if atom.GetSymbol() == "*":
                        rm_ids.append(atom.GetIdx())

                rm_ids.sort(reverse=True)
                rwmol = RWMol(frag_mol)
                for i in rm_ids:
                    rwmol.RemoveAtom(i)

                frag_mol = rwmol.GetMol()
                vocab_list.append(Chem.MolToSmiles(frag_mol))
        elif parse_type == "ring":
            frag_smiles = parse_ring(mol)
            for fs in frag_smiles.split("."):
                fmol = Chem.MolFromSmiles(fs)
                fmol = clear_atommap(fmol)
                vocab_list.append(Chem.MolToSmiles(fmol, isomericSmiles=False))

    vocab_list = collections.Counter(vocab_list)
    # with open(out_path+f"ver2/vocab-{parse_type}-ver2.pickle", mode="wb") as f:
    #     pickle.dump(vocab_list, f)

    vocab_list = list(set(vocab_list))
    with open(out_path+f"ver2/vocab-{parse_type}-ver2.smi", mode="w") as f:
        for smiles in vocab_list:
            f.write(smiles+"\n")


def make_features(n, outpath="data/features/ver2/"):
    fragment_list = read_smilesset("data/features/ver2/vocab-ring-ver2.smi")[:n]

    graph_list = []
    label_list = []
    num_node_list = []

    for n, smiles in enumerate(tqdm(fragment_list)):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol = setBFSorder(mol)

        for n_atom in range(mol.GetNumAtoms()):
            rm_id_list = []
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() > n_atom+1:
                    rm_id_list.append(atom.GetIdx())

            rwmol = RWMol(mol)
            rm_id_list = sorted(rm_id_list, reverse=True)
            for rm_id in rm_id_list:
                rwmol.RemoveAtom(rm_id)

            base_mol = rwmol.GetMol()
            # print(Chem.MolToSmiles(base_mol))
            graph = Mol2Graph(Chem.MolFromSmiles(Chem.MolToSmiles(base_mol)))

            label_atom = len(ATOM_IDX)
            label_bond = [len(BOND_IDX)+1] * MAX_NODE
            for i in range(base_mol.GetNumAtoms()):
                label_bond[i] = 0
            if n_atom < mol.GetNumAtoms()-1:
                atom = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() == n_atom+2][0]
                label_atom = ATOM_IDX[atom.GetSymbol()]
                for nei_atom in atom.GetNeighbors():
                    if nei_atom.GetAtomMapNum() < atom.GetAtomMapNum():
                        bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), nei_atom.GetIdx()).GetBondType()
                        label_bond[base_mol.GetNumAtoms()-nei_atom.GetAtomMapNum()] = RDKIT_BOND[bond_type]

            graph_list.append(graph)
            label_list.append({"atom": label_atom, "bond": label_bond})
            num_node_list.append(base_mol.GetNumAtoms())

    print(len(graph_list))

    with open(outpath+"dgl-graph.pickle", mode="wb") as f:
        pickle.dump(graph_list, f)
    with open(outpath+"label.pickle", mode="wb") as f:
        pickle.dump(label_list, f)
    with open(outpath+"num-node.pickle", mode="wb") as f:
        pickle.dump(num_node_list, f)


def swap_mid_tobfs(mol, mids):
    copied_mol = deepcopy(mol)
    att_atom_id = -1
    att_atom_mid = -1
    for atom in copied_mol.GetAtoms():
        if atom.GetAtomMapNum() in mids:
            att_atom_id = atom.GetIdx()
    copied_mol = setBFSorder(copied_mol)
    for atom in copied_mol.GetAtoms():
        if atom.GetIdx() == att_atom_id:
            att_atom_mid = atom.GetAtomMapNum()

    return att_atom_mid


def vocab_test():
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open("../data/vocabulary/vocab-ring.pickle", mode="rb") as f:
        vocab = pickle.load(f)

    print(len(vocab))

    num_atom = []
    for smiles in tqdm(vocab):
        mol = Chem.MolFromSmiles(smiles)
        num_atom.append(mol.GetNumAtoms())

    sns.distplot(num_atom, kde=False)
    plt.show()


def main():
    # make_vocabulary(parse_type="ring")
    make_features(n=100000, outpath="data/features/ver2/")


if __name__ == '__main__':
    main()
