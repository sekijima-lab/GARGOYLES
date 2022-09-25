import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import read_smilesset
from Utils.mol_utils import setBFSorder, valence_mask, add_frag, clear_atommap, ATOM_IDX, parse_ring, update_atom_rm
from Utils.reward import getReward
from Generator.sampling import Sampler

import math
import numpy as np
import pandas as pd
import pickle
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, rdmolops, RWMol, DataStructs
from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')


class AtomNode:
    def __init__(self, smiles, att_mid=-1):
        self.parent = None
        self.smiles = smiles
        self.depth = 0
        self.visit = 0
        self.children = []
        self.imm_score = 0
        self.cum_score = 0
        self.c = 1
        self.id = -1
        self.rollout_result = ("None", -1000)
        self.att_mid = att_mid

    def add_Node(self, c):
        c.parent = self
        c.depth = self.depth + 1
        self.children.append(c)

    def calc_UCB(self):
        if self.visit == 0:
            ucb = 1e+6
        else:
            ucb = self.cum_score/self.visit + self.c*math.sqrt(2*math.log(self.parent.visit)/self.visit)
        return ucb

    def select_children(self):
        children_ucb = []
        for cn in self.children:
            children_ucb.append(cn.calc_UCB())
        max_ind = np.random.choice(np.where(np.array(children_ucb) == max(children_ucb))[0])
        return self.children[max_ind]


class MolNode:
    def __init__(self, smiles, att_mid=-1):
        self.parent = None
        self.smiles = smiles
        self.depth = 0
        self.visit = 0
        self.children = []
        self.imm_score = 0
        self.cum_score = 0
        self.c = 1
        self.id = -1
        self.rollout_result = ("None", -1000)
        self.att_mid = att_mid

    def add_Node(self, c):
        c.parent = self
        c.depth = self.depth + 1
        self.children.append(c)

    def calc_UCB(self):
        if self.visit == 0:
            ucb = 1e+6
        else:
            ucb = self.cum_score/self.visit + self.c*math.sqrt(2*math.log(self.parent.visit)/self.visit)
        return ucb

    def select_children(self):
        children_ucb = []
        for cn in self.children:
            children_ucb.append(cn.calc_UCB())
        max_ind = np.random.choice(np.where(np.array(children_ucb) == max(children_ucb))[0])
        return self.children[max_ind]


class CoarseMCTS:
    def __init__(self, seed_smiles, reward_name, sampler, max_score=-1e+5, th=0.):
        self.root = AtomNode(smiles=seed_smiles)
        self.sampler = sampler
        self.current_node = None
        self.next_node = []
        self.rollout_result = {}
        self.total_nodes = 0
        self.step = 0
        self.th = th
        self.max_score = max_score
        self.generated_smiles = pd.DataFrame()
        self.max_seq = 10
        self.reward_module = getReward(name=reward_name, seed_smiles=seed_smiles, th=th)
        self.reward_name = reward_name
        self.init_score = self.reward_module.reward(Chem.MolFromSmiles(seed_smiles))

    def set_state(self, node):
        seed_mol = Chem.MolFromSmiles(node.smiles)
        seed_mol = setBFSorder(seed_mol)
        # print(Chem.MolToSmiles(seed_mol))

        for bond in seed_mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            rwmol = RWMol(seed_mol)
            begin_atom = bond.GetBeginAtom()
            begin_atom = rwmol.GetAtomWithIdx(begin_atom.GetIdx())
            end_atom = bond.GetEndAtom()
            end_atom = rwmol.GetAtomWithIdx(end_atom.GetIdx())
            att_mids = [begin_atom.GetAtomMapNum(), end_atom.GetAtomMapNum()]
            rwmol.RemoveBond(begin_atom.GetIdx(), end_atom.GetIdx())

            update_atom_rm(begin_atom)
            update_atom_rm(end_atom)

            out_mol = rwmol.GetMol()
            smiles = Chem.MolToSmiles(out_mol)
            if "." in smiles:
                s1, s2 = smiles.split(".")
                mol1 = Chem.MolFromSmiles(s1)
                mol2 = Chem.MolFromSmiles(s2)
                if mol1.GetNumAtoms() < mol2.GetNumAtoms():
                    mol1, mol2 = mol2, mol1

                for atom in mol1.GetAtoms():
                    if atom.GetAtomMapNum() in att_mids:
                        print(Chem.MolToSmiles(clear_atommap(mol1)))
                        cnode = MolNode(smiles=Chem.MolToSmiles(mol1))
                        node.add_Node(cnode)

        # Add
        cnode = MolNode(smiles=Chem.MolToSmiles(seed_mol))
        node.add_Node(cnode)
        # mask = valence_mask(mol=seed_mol)
        # smiles = Chem.MolToSmiles(seed_mol)
        # for i in range(len(mask)):
        #     if mask[i] > 0:
        #         cnode = MolNode(smiles=smiles)
        #         node.add_Node(cnode)

    def _select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()

                self.current_node = self.root

    def _simulate(self, n_step, num_next_state=3):
        print("--- Sim ---")
        fine_mcts = FineMCTS(seed_smiles=self.current_node.smiles, reward_module=self.reward_module,
                             sampler=self.sampler)
        fine_mcts.search(n_step=n_step, start_step=self.step)
        self.max_score = max(self.max_score, fine_mcts.max_score)

        df = pd.DataFrame()
        df["SMILES"] = fine_mcts.generated_smiles["SMILES"]
        df["Reward"] = fine_mcts.generated_smiles["Reward"]
        df["Step"] = fine_mcts.generated_smiles["Step"]
        df["Imp"] = df["Reward"] - self.reward_module.reward(Chem.MolFromSmiles(self.current_node.smiles))
        self.generated_smiles = pd.concat([self.generated_smiles, df])
        df = df[df["Imp"] > 0]

        if len(df) > 0:
            next_states = self._select_next_state(df)
            for smiles, value in zip(next_states["SMILES"].to_list(), next_states["Reward"].to_list()):
                cnode = MolNode(smiles=smiles)
                self.set_state(cnode)
                self.current_node.add_Node(cnode)

        # for cnode in self.current_node.children:
        #     if Chem.MolFromSmiles(cnode.smiles) is None:
        #         continue
        #
        #     rollout_result = self.sampler.sample(smiles=cnode.smiles, n=5)
        #     if len(rollout_result) == 0:
        #         cnode.imm_score = -10000
        #         cnode.cum_score = -10000
        #         self.current_node.visit = 10000
        #     else:
        #         mol_list = self._add_fragment(cnode, frag_smiles=rollout_result[0])
        #         reward_list = []
        #
        #         if mol_list is not None and len(mol_list) > 0:
        #             mol_list = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in mol_list]
        #             mol_list = [mol for mol in mol_list if mol is not None]
        #             for mol in mol_list:
        #                 reward_list.append(self.reward_module.reward(mol))
        #             if len(mol_list) == 0:
        #                 continue
        #             max_id = np.argmax(reward_list)
        #             if reward_list[max_id] > self.reward_module.vmin:
        #                 max_reward = reward_list[max_id]
        #                 self.max_score = max(self.max_score, max_reward)
        #                 cnode.imm_score = max_reward/(1+abs(max_reward))
        #                 cnode.cum_score = max_reward/(1+abs(max_reward))
        #                 self.generated_smiles["SMILES"].append(Chem.MolToSmiles(clear_atommap(mol_list[max_id])))
        #                 self.generated_smiles["Reward"].append(reward_list[max_id])
        #                 self.generated_smiles["Step"].append(self.step)
        #                 print("Reward: ", reward_list[max_id])
        #                 print(self.generated_smiles["SMILES"][-1])

    def _select_next_state(self, df):
        next_states = df.sample(n=min(len(df), 3))
        next_states.append(df.iloc[0])

        return next_states.drop_duplicates()

    def _add_fragment(self, node, frag_smiles):
        if Chem.MolFromSmiles(frag_smiles) is None:
            return None

        cnode = node
        while cnode.parent.depth != 0:
            cnode = cnode.parent

        mol_list = add_frag(Chem.MolFromSmiles(cnode.smiles), att_mid=cnode.att_mid, frag_smiles=frag_smiles)

        return mol_list

    def _backprop(self):
        reward_list = []
        for child_node in self.current_node.children:
            reward_list.append(child_node.imm_score)

        max_reward = -10000
        if len(reward_list) > 0:
            max_reward = max(reward_list)
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward/(1+abs(max_reward)))
            self.current_node = self.current_node.parent

    def search(self, n_step_coarse, n_step_fine, start_step=0):
        self.set_state(self.root)
        self.step = start_step
        while self.step < n_step_coarse+start_step:
            # 1 Selection
            self._select()
            print("depth: ", self.current_node.depth)

            # 3 Simulation
            self._simulate(n_step=n_step_fine)

            # 4 Backpropagation
            self._backprop()

            self.step += n_step_fine
            print("--- Coarse MCTS step %d ---" % self.step)
            print("MAX_SCORE:", self.max_score, self.max_score - self.init_score)


class FineMCTS:
    def __init__(self, seed_smiles, reward_module, sampler, max_score=-1e+5):
        self.root = AtomNode(smiles=seed_smiles)
        self.sampler = sampler
        self.current_node = None
        self.next_token = {}
        self.rollout_result = {}
        self.total_nodes = 0
        self.step = 0
        self.max_score = max_score
        self.generated_smiles = {"SMILES": [], "Reward": [], "Step": []}
        self.max_seq = 10
        self.reward_module = reward_module
        self.init_score = self.reward_module.reward(Chem.MolFromSmiles(seed_smiles))

    def set_initial_state(self):
        mol = Chem.MolFromSmiles(self.root.smiles)
        mol = setBFSorder(mol)
        mask = valence_mask(mol)
        for i in range(len(mask)):
            if mask[i] == 0:
                continue

            node = AtomNode(smiles="<start>", att_mid=i+1)
            node.add_Node(AtomNode(smiles="C", att_mid=i+1))
            node.add_Node(AtomNode(smiles="O", att_mid=i+1))
            node.add_Node(AtomNode(smiles="N", att_mid=i+1))
            self.root.add_Node(node)

    def _select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent

                self.current_node = self.root

    def _expand(self, n):
        print("--- Exp ---")
        print("step: ", self.current_node.smiles)
        print("depth: ", self.current_node.depth)
        current_mol = Chem.MolFromSmiles(self.current_node.smiles)

        expand_smiles_list = []
        if current_mol is not None:
            pred_atom_list, pred_edge_list = self.sampler.step(current_mol, n=n)

            for i in range(n):
                if pred_atom_list[i] == len(ATOM_IDX):
                    # *****
                    # Processing when a termination symbol is found

                    # *****
                    continue
                mol = self.sampler.revise_mol(current_mol, [pred_atom_list[i]], [pred_edge_list[i]])
                if mol is not None:
                    expand_smiles_list.append(Chem.MolToSmiles(clear_atommap(mol[0])))
                    # print("Add Node: ", Chem.MolToSmiles(current_mol), Chem.MolToSmiles(clear_atommap(mol[0])))

            expand_smiles_list = list(set(expand_smiles_list))
            for smiles in expand_smiles_list:
                node = AtomNode(smiles=smiles, att_mid=self.current_node.att_mid)
                node.imm_score = -10000
                node.cum_score = -10000
                node.visit = 1
                self.current_node.add_Node(node)

        if current_mol is None or len(expand_smiles_list) == 0:
            # Discard this node because the SMILES corresponding to the current node is invalid
            self.current_node.imm_score = -10000
            self.current_node.cum_score = -10000
            self.current_node.visit = 10000

        # print([cnode.smiles for cnode in self.current_node.children])

    def _simulate(self):
        # print("--- Sim ---")
        for cnode in self.current_node.children:
            if Chem.MolFromSmiles(cnode.smiles) is None:
                continue

            rollout_result = self.sampler.sample(smiles=cnode.smiles, n=10)
            if len(rollout_result) == 0:
                # No valid molecules are produced from this child node
                cnode.imm_score = -10000
                cnode.cum_score = -10000
                cnode.visit = 10000
            else:
                mol_list = self._add_fragment(frag_smiles=rollout_result[0], att_mid=self.current_node.att_mid)
                reward_list = []

                if mol_list is not None and len(mol_list) > 0:
                    mol_list = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in mol_list]
                    mol_list = [mol for mol in mol_list if mol is not None]
                    if len(mol_list) == 0:
                        cnode.imm_score = -10000
                        cnode.cum_score = -10000
                        cnode.visit = 10000
                        continue
                    for mol in mol_list:
                        reward_list.append(self.reward_module.reward(mol))
                    max_id = np.argmax(reward_list)
                    if reward_list[max_id] > self.reward_module.vmin:
                        max_reward = reward_list[max_id]
                        self.max_score = max(self.max_score, max_reward)
                        cnode.imm_score = max_reward/(1+abs(max_reward))
                        cnode.cum_score = max_reward/(1+abs(max_reward))
                        self.generated_smiles["SMILES"].append(Chem.MolToSmiles(clear_atommap(mol_list[max_id])))
                        self.generated_smiles["Reward"].append(reward_list[max_id])
                        self.generated_smiles["Step"].append(self.step)
                        print("Reward: ", reward_list[max_id])
                        print(self.generated_smiles["SMILES"][-1])

    def _add_fragment(self, frag_smiles, att_mid):
        mol = Chem.MolFromSmiles(self.root.smiles)
        mol = setBFSorder(mol)
        mol_list = add_frag(mol, att_mid=att_mid, frag_smiles=frag_smiles)
        clear_atommap(mol)

        return mol_list

    def _backprop(self):
        max_reward = max([cnode.imm_score for cnode in self.current_node.children])
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward/(1+abs(max_reward)))
            self.current_node = self.current_node.parent

    def search(self, n_step, start_step=0):
        self.set_initial_state()
        self.step = start_step

        while self.step < n_step+start_step:
            self.step += 1

            # 1 Selection
            self._select()
            # print("depth: ", self.current_node.depth)

            # 2 Expand
            self._expand(n=5)

            if len(self.current_node.children) > 0:
                # 3 Simulation
                self._simulate()

                # 4 Backpropagation
                self._backprop()

            print("---Fine MCTS step %d ---" % self.step)
            print("MAX_SCORE:", self.max_score, self.max_score - self.init_score)


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    smiles_list = read_smilesset(hydra.utils.get_original_cwd()+"/data/zinc_250k.smi")
    sampler = Sampler(cfg)
    reward_module = getReward(name="QED")

    # nums = []
    # from tqdm import tqdm
    # for smiles in tqdm(smiles_list):
    #     # smiles = "Cc1sc(=O)n(CC(=O)N2CCC(Nc3ccccc3)CC2)c1-c1ccc(F)cc1"
    #     # mcts = FineMCTS(seed_smiles=smiles, reward_module=reward_module, sampler="sampler")
    #     mcts = CoarseMCTS(seed_smiles=smiles, reward_name="QED", sampler="")
    #     mcts.set_state(mcts.root)
    #     nums.append(len(mcts.root.children))
    #
    #     # for node in mcts.root.children:
    #     #     print(node.smiles)
    #     # break
    #
    # print(np.mean(nums), np.std(nums))

    smiles = "O=c1n(CCO)c2ccccc2n1CCO"
    # mcts = FineMCTS(seed_smiles=smiles, reward_module=reward_module, sampler=sampler)
    mcts = CoarseMCTS(seed_smiles=smiles, reward_name="QED", sampler=sampler)
    mcts.search(n_step_coarse=2000, n_step_fine=100)

    df = pd.DataFrame()

    sim_list = []
    seed_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2)
    for gent_smiles in mcts.generated_smiles["SMILES"]:
        gent_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(gent_smiles), 2)
        sim = DataStructs.TanimotoSimilarity(seed_fp, gent_fp)
        sim_list.append(sim)

    df["SMILES"] = mcts.generated_smiles["SMILES"]
    df["Reward"] = mcts.generated_smiles["Reward"]
    df["Sim"] = sim_list
    df["Step"] = mcts.generated_smiles["Step"]
    df["Imp"] = df["Reward"] - reward_module.reward(Chem.MolFromSmiles(smiles))
    df = df.sort_values("Reward", ascending=False)
    df.to_csv(hydra.utils.get_original_cwd()+"/data/result/sample.csv", index=False)


if __name__ == '__main__':
    main()

