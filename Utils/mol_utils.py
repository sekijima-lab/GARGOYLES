import copy
import os
import numpy as np

import torch

from dgllife.utils import mol_to_bigraph

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, QED, DataStructs, BRICS, Descriptors, Crippen, rdmolops
from rdkit.Chem.rdchem import RWMol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ATOM_IDX = {"C": 0, "N": 1, "O": 2, "F": 3, "P": 4, "S": 5, "Cl": 6, "Br": 7, "I": 8}
ATOM_IDX_INV = {0: "C", 1: "N", 2: "O", 3: "F", 4: "P", 5: "S", 6: "Cl", 7: "Br", 8: "I"}
PERIODIC_TABLE = {"C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}
VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1}

# BOND_IDX = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMA": 4}
# RDKIT_BOND = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
# RDKIT_BOND_INV = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
BOND_IDX = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
RDKIT_BOND = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
RDKIT_BOND_INV = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

# VOCABULARY = [
#     'PAD',
#     '#', '(', ')', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=',
#     'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S',
#     '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH-]', '[CH2-]',
#     '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[NH4+]', '[NH2-]',
#     '[O+]', '[O-]', '[OH+]', '[OH-]',
#     '[P+]', '[P@@H]', '[P@@]', '[P@]', '[PH+]', '[PH2]', '[PH]',
#     '[S+]', '[S-]', '[S@@+]', '[S@@]', '[S@]', '[SH+]', '[SH]', '[SH-]',
#     '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '\\', 'c', 'n', 'o', 's',
#     '&', '\n'
# ]

VOCABULARY = ['[S@]', '4', 'C', '[C@]', '[n+]', '\\', '[S-]', '[O-]', '2', '[OH-]', '1', '(', '&', '5', '3', '[C@H]',
              '[NH+]', '/', '\n', '[S@@]', '[SH-]', 'O', '[nH]', '[SH]', 'Cl', 'S', '[C@@]', 'F', '#', '-', 'c', '[N-]',
              '=', '[C@@H]', '[nH+]', 'N', 'n', 's', '[NH3+]', ')', '[NH2+]', '[N+]', 'I', 'Br', 'o', '[NH-]', '[NH4+]']
VOCABULARY.append("&")
VOCABULARY.append("\n")
VOCABULARY.insert(0, "PAD")

MAX_NODE = 38
MAX_SEQ_LENGTH = 40


def parse_ring(mol):
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    rm_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.IsInRing() and not atom2.IsInRing() or not atom1.IsInRing() and atom2.IsInRing():
                rm_bonds.append((atom1.GetAtomMapNum(), atom2.GetAtomMapNum()))
            # elif not atom1.IsInRing() and not atom2.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            #     rm_bonds.append((atom1.GetAtomMapNum(), atom2.GetAtomMapNum()))
            elif atom1.IsInRing() and atom2.IsInRing():
                # Remove the bond between two rings.
                is_samering = False
                for ri_idxs in rings:
                    if atom1.GetIdx() in ri_idxs and atom2.GetIdx() in ri_idxs:
                        is_samering = True
                if not is_samering:
                    rm_bonds.append((atom1.GetAtomMapNum(), atom2.GetAtomMapNum()))

    rwmol = RWMol(mol)
    for i, j in rm_bonds:
        atom1 = [atom for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == i][0]
        atom2 = [atom for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == j][0]

        update_atom_rm(atom1)
        update_atom_rm(atom2)
        rwmol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())

    mol = rwmol.GetMol()

    return Chem.MolToSmiles(mol)


def update_atom_rm(atom):
    if atom.GetFormalCharge() == 1:
        atom.SetFormalCharge(0)
    elif atom.GetFormalCharge() == -1:
        atom.SetFormalCharge(0)
    else:
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)


def add_frag(mol, att_mid, frag_smiles):
    # print(Chem.MolToSmiles(mol))
    # print(frag_smiles)
    # print(att_mid)
    out_mol_list = []
    frag_mol = Chem.MolFromSmiles(frag_smiles)
    rwmol = RWMol(mol)
    atom = [atom for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == att_mid][0]
    if atom.GetSymbol() == "S":
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
    elif atom.GetNumExplicitHs() > 0:
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
    new_id = rwmol.AddAtom(Chem.Atom("*"))
    rwmol.AddBond(atom.GetIdx(), new_id, Chem.BondType.SINGLE)
    for i in range(frag_mol.GetNumAtoms()):
        outmol = AllChem.ReplaceSubstructs(rwmol.GetMol(), Chem.MolFromSmiles("*"), frag_mol,
                                           replacementConnectionPoint=i)[0]
        try:
            outmol.UpdatePropertyCache(strict=True)
        except:
            continue
        out_mol_list.append(outmol)

    return out_mol_list


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


def clear_atommap(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def valence_mask(mol):
    mask = [0]*mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        valence = 0
        for natom in atom.GetNeighbors():
            btype = mol.GetBondBetweenAtoms(atom.GetIdx(), natom.GetIdx()).GetBondType()
            if btype == Chem.BondType.AROMATIC:
                valence += 1.5
            else:
                valence += RDKIT_BOND_INV[btype]

        if valence < VALENCY[atom.GetSymbol()]:
            mask[atom.GetAtomMapNum()-1] = 1

    return mask


def Mol2Graph(mol):
    # Sort the atomic orders in Mol object
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = setBFSorder(mol)
    new_order = [-1] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        new_order[atom.GetAtomMapNum() - 1] = atom.GetIdx()
    mol = rdmolops.RenumberAtoms(mol, new_order)
    g = mol_to_bigraph(mol, node_featurizer=featurize_atoms, edge_featurizer=featurize_bonds)

    return g


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(ATOM_IDX[atom.GetSymbol()])
    return {'atomic': torch.tensor(feats, dtype=torch.long).reshape(-1, )}


def featurize_bonds(mol):
    feats = []
    for bond in mol.GetBonds():
        btype = RDKIT_BOND[bond.GetBondType()]-1
        feats.extend([btype, btype])

    return {'type': torch.tensor(feats, dtype=torch.long).reshape(-1, )}


def Mol2Mat(mol, sub_feature=False):
    adj = np.zeros([MAX_NODE, MAX_NODE])
    if sub_feature:
        node_feat = np.zeros([MAX_NODE, len(ATOM_IDX)+7+2])  # atom type + valency + is in ring?
    else:
        node_feat = np.zeros([MAX_NODE, len(ATOM_IDX)])

    for atom in mol.GetAtoms():
        map_id = atom.GetAtomMapNum()
        node_feat[map_id-1, ATOM_IDX[atom.GetSymbol()]] = 1
        if sub_feature:
            node_feat[map_id-1, len(ATOM_IDX) + atom.GetExplicitValence()] = 1
            node_feat[map_id-1, len(ATOM_IDX) + 7 + int(atom.IsInRing())] = 1

    for bond in mol.GetBonds():
        from_map_id = bond.GetBeginAtom().GetAtomMapNum()
        to_map_id = bond.GetEndAtom().GetAtomMapNum()
        btype = RDKIT_BOND[bond.GetBondType()]

        adj[from_map_id-1, to_map_id-1] = btype
        adj[to_map_id-1, from_map_id-1] = btype

    return adj, node_feat


def Mat2Mol(adj, node_feat):
    num_node = int(sum((sum(node_feat))))
    # node_feat = np.identity(node_feat.shape[1])[node_feat]
    rwmol = RWMol()

    for i in range(num_node):
        atom = ATOM_IDX_INV[int(np.argmax(node_feat[i]))]
        from_idx = rwmol.AddAtom(Chem.Atom(atom))
        rwmol.GetAtomWithIdx(from_idx).SetAtomMapNum(i+1)

    for i in range(num_node):
        adj_col = adj[i, i+1:num_node]
        from_idx = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == i+1][0]
        for j in range(len(adj_col)):
            if adj_col[j] != 0:
                btype = RDKIT_BOND_INV[adj_col[j]]
                to_idx = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == (i+1)+(j+1)][0]
                rwmol.AddBond(from_idx, to_idx, btype)

    for atom in rwmol.GetAtoms():
        valence = 0
        for natom in atom.GetNeighbors():
            btype = rwmol.GetBondBetweenAtoms(atom.GetIdx(), natom.GetIdx()).GetBondType()
            if btype == Chem.BondType.AROMATIC:
                valence += 1.5
            else:
                valence += RDKIT_BOND_INV[btype]

        if atom.GetSymbol() == "N":
            if valence >= 4:
                atom.SetFormalCharge(1)
            elif valence == 3 and atom.GetFormalCharge() == 1:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() == "O":
            if valence == 1:
                atom.SetNumExplicitHs(1)
            elif valence == 2:
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == "S":
            if valence == 3 and atom.GetIsAromatic():
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == "C":
            if valence == 4 and atom.GetNumExplicitHs() == 1:
                atom.SetNumExplicitHs(0)

    rwmol.UpdatePropertyCache(strict=True)
    mol = rwmol.GetMol()

    return mol


def main():
    smiles = "c1c(F)c(F)c(F)cc1"
    mol = Chem.MolFromSmiles(smiles)
    mol = setBFSorder(mol)
    # out_mol = add_frag(mol, 1, "Cl")
    out_mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles("F"), Chem.MolFromSmiles("OC"),
                                        replacementConnectionPoint=1)

    for m in out_mol:
        print(Chem.MolToSmiles(clear_atommap(m)))


if __name__ == '__main__':
    main()
