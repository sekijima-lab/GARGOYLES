"""Single-objective MCTS with corrected reward backup and bounded execution.

Coarse normalizes raw simulation rewards once; Fine backs up values that
were already normalized during rollout. See docs/mcts.md for usage.
"""
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

import copy
from functools import reduce
import json
from scipy import spatial
from pathlib import Path
from Generator.search_budget import (BudgetedReward, SearchBudgetExceeded,
    check_budget, install_request_timeout)

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
        c._set_depth(self.depth + 1)
        self.children.append(c)

    def _set_depth(self, depth):
        self.depth = depth
        for child in self.children:
            child._set_depth(depth + 1)

    def remove_Node(self):
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None

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
        c._set_depth(self.depth + 1)
        self.children.append(c)

    def _set_depth(self, depth):
        self.depth = depth
        for child in self.children:
            child._set_depth(depth + 1)

    def remove_Node(self):
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None

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
    def __init__(self, seed_smiles, reward_module, sampler, max_score=-1e+5, th=0.):
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
        self.reward_module = reward_module #getReward(name=reward_name, seed_smiles=seed_smiles, th=th)
        #self.reward_name = reward_name
        self.init_score = self.reward_module.reward(Chem.MolFromSmiles(seed_smiles))
        self.completed_steps = 0
        self.stop_reason = "step_budget"
        self.checkpoint = None
        self.coarse_updates = []

    def set_state(self, node):
        if isinstance(node.smiles,str):
            seed_mol = Chem.MolFromSmiles(node.smiles)
            seed_mol = setBFSorder(seed_mol)
        else:
            return
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
        self.current_node = self.root
        while self.current_node.children:
            check_budget()
            self.current_node = self.current_node.select_children()
            if self.current_node.depth + 1 > self.max_seq:
                rejected = self.current_node
                node = rejected
                while node is not None:
                    node.cum_score -= 1
                    node.visit += 1
                    node = node.parent
                rejected.remove_Node()
                self.current_node = self.root


    def _simulate(self, n_step, num_next_state=3):
        check_budget()
        if not isinstance(self.current_node.smiles, str):
            raise ValueError("Coarse node must contain a SMILES string")
        fine_mcts = FineMCTS(seed_smiles=self.current_node.smiles,
            reward_module=self.reward_module, sampler=self.sampler)
        try:
            fine_mcts.search(n_step=n_step, start_step=self.step)
        except SearchBudgetExceeded:
            self.stop_reason = "walltime_budget"
        self.completed_steps = fine_mcts.completed_steps
        self.max_score = max(self.max_score, fine_mcts.max_score)
        df = pd.DataFrame(fine_mcts.generated_smiles)
        df["Imp"] = df["Reward"] - fine_mcts.init_score
        df["FineStepComplete"] = df["Step"] <= fine_mcts.completed_steps
        self.generated_smiles = pd.concat([self.generated_smiles, df], ignore_index=True)
        improved = df[df["Imp"] > 0]
        if improved.empty:
            # An interrupted simulation is not evidence of chemical failure.
            return None if self.stop_reason == "walltime_budget" else -10000.0
        simulation_reward = float(improved["Reward"].max())
        next_states = self._select_next_state(improved, num_next_state)
        for row in next_states.itertuples(index=False):
            cnode = MolNode(smiles=row.SMILES)
            cnode.raw_reward = float(row.Reward)
            self.current_node.add_Node(cnode)
            self.set_state(cnode)
        return simulation_reward


    def _select_next_state(self, df, num_next_state=3):
        valid = df[df["SMILES"].map(lambda s: isinstance(s, str))].copy()
        valid = valid[np.isfinite(valid["Reward"])].drop_duplicates("SMILES")
        if valid.empty or num_next_state < 1:
            return valid.iloc[:0]
        elite = valid.nlargest(1, "Reward")
        rest = valid.drop(elite.index)
        sampled = rest.sample(n=min(len(rest), num_next_state - 1))
        return pd.concat([elite, sampled], ignore_index=True)


    def _add_fragment(self, node, frag_smiles):
        if Chem.MolFromSmiles(frag_smiles) is None:
            return None

        cnode = node
        while cnode.parent.depth != 0:
            cnode = cnode.parent

        mol_list = add_frag(Chem.MolFromSmiles(cnode.smiles), att_mid=cnode.att_mid, frag_smiles=frag_smiles)

        return mol_list

    def _backprop(self, raw_reward):
        if raw_reward is None:
            return
        score = raw_reward / (1 + abs(raw_reward))
        self.coarse_updates.append({"step": self.completed_steps,
            "selected_smiles": self.current_node.smiles,
            "raw_reward": raw_reward, "backup_score": score,
            "depth": self.current_node.depth})
        while self.current_node is not None:
            node = self.current_node
            node.imm_score = score if node.visit == 0 else max(node.imm_score, score)
            node.visit += 1
            node.cum_score += score
            self.current_node = node.parent


    def search(self, n_step_coarse, n_step_fine, start_step=0):
        if n_step_fine <= 0:
            raise ValueError("n_step_fine must be positive")
        self.set_state(self.root)
        self.step = start_step
        self.completed_steps = start_step
        target_step = start_step + n_step_coarse
        try:
            while self.completed_steps < target_step:
                check_budget()
                self.step = self.completed_steps
                self._select()
                reward = self._simulate(min(n_step_fine, target_step - self.step))
                self._backprop(reward)
                self.step = self.completed_steps
                if self.checkpoint:
                    self.checkpoint(self)
                print(f"Coarse completed_steps={self.completed_steps} max_reward={self.max_score}", flush=True)
                if self.stop_reason == "walltime_budget":
                    break
        except SearchBudgetExceeded:
            self.stop_reason = "walltime_budget"
        finally:
            if self.checkpoint:
                self.checkpoint(self)



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
        self.completed_steps = 0

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
        self.current_node = self.root
        while self.current_node.children:
            check_budget()
            self.current_node = self.current_node.select_children()
            if self.current_node.depth + 1 > self.max_seq:
                rejected = self.current_node
                node = rejected
                while node is not None:
                    node.cum_score -= 1
                    node.visit += 1
                    node = node.parent
                rejected.remove_Node()
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
        # Simulation stored g(R), so back it up without applying g again.
        score = max(cnode.imm_score for cnode in self.current_node.children)
        score = max(-1.0, min(1.0, score))  # bound legacy invalid-rollout sentinels
        while self.current_node is not None:
            node = self.current_node
            node.imm_score = score if node.visit == 0 else max(node.imm_score, score)
            node.visit += 1
            node.cum_score += score
            self.current_node = node.parent


    def search(self, n_step, start_step=0):
        self.set_initial_state()
        self.step = start_step
        self.completed_steps = start_step
        while self.completed_steps < n_step + start_step:
            check_budget()
            self.step = self.completed_steps + 1
            self._select()
            self._expand(n=5)
            if self.current_node.children:
                self._simulate()
                self._backprop()
            self.completed_steps = self.step
            print(f"Fine completed_step={self.completed_steps} max_reward={self.max_score}", flush=True)


class Pareto():
    """
    Pareto Front Class
    """

    def __init__(self,front=None,cfg:DictConfig=None) -> None:
        """
        Constructor

        Input:
            - front: reward score vectors in pareto front
        """
        self.CONFIG = cfg
        self.front = front if front is not None else [[0 for _ in range(len(self.CONFIG["reward"]["reward_list"]))]]

    @classmethod
    def filter_(cls, pts, pt):
        """
        Get all points in pts that are not Pareto dominated by the point pt
        """
        weakly_worse   = (pts <= pt).all(axis=-1)
        strictly_worse = (pts < pt).any(axis=-1)
        return pts[~(weakly_worse & strictly_worse)]

    @classmethod
    def get_pareto_undominated_by(cls, pts1, pts2=None):
        """
        Return all points in pts1 that are not Pareto dominated
        by any points in pts2
        """
        if pts2 is None:
            pts2 = pts1
        return reduce(Pareto.filter_, pts2, pts1)

    @classmethod
    def get_pareto_frontier(cls, pts)->np.ndarray:
        """
        Iteratively filter points based on the convex hull heuristic
        """
        pareto_groups = []

        # loop while there are points remaining
        while pts.shape[0]:
            # brute force if there are few points:
            if pts.shape[0] < 10:
                pareto_groups.append(Pareto.get_pareto_undominated_by(pts))
                break

            # compute vertices of the convex hull
            hull_vertices = spatial.ConvexHull(pts).vertices

            # get corresponding points
            hull_pts = pts[hull_vertices]

            # get points in pts that are not convex hull vertices
            nonhull_mask = np.ones(pts.shape[0], dtype=bool)
            nonhull_mask[hull_vertices] = False
            pts = pts[nonhull_mask]

            # get points in the convex hull that are on the Pareto frontier
            pareto   = Pareto.get_pareto_undominated_by(hull_pts)
            pareto_groups.append(pareto)

            # filter remaining points to keep those not dominated by
            # Pareto points of the convex hull
            pts = Pareto.get_pareto_undominated_by(pts, pareto)

        return np.vstack(pareto_groups)

    def __get_new_tmp_pareto(self,m:list)->list:
        
        __tmp_front = copy.deepcopy(self.front)
        __tmp_front.append(m)
        __tmp_front = Pareto.get_pareto_undominated_by(np.array(__tmp_front)).tolist()
        return __tmp_front


    def dominated(self, m:list)->bool:
        """Is point m dominated in reward space?
        
            Input
                - m: reward vector to decide dominated or not: List type

            Return
                - False: Dominated
                - True: Non-dominated (in pareto front)
        """
        return m in self.__get_new_tmp_pareto(m)
    
    def update(self,scores:list)->None:
        """
        Update Pareto Front
        Input:
            - scores: reward vector
        Return:
            - None
        """
        self.front = self.__get_new_tmp_pareto(scores)
        with open(self.CONFIG["mcts"]["out_dir"]+"present/output.txt", 'a') as f:
            f.write(f"pareto size:{len(self.front)}\n")
            f.write(f"Updated pareto front\n{self.front}\n")
            f.write(f"Time:{time.asctime(time.localtime(time.time()))}\n")
        print(f"pareto size:{len(self.front)}")

    @staticmethod
    def from_dict(_filename, cfg: DictConfig):
        """Pareto front backup from files
            WARNING: you should check _filename file exists
            Input:
                - _filename: filepath of pareto.json

            Return:
                - Initialized Pareto front
        """
        with open(_filename, 'r') as f:
            _set_json = json.load(f)
            new_pareto = Pareto(front= _set_json['front'], cfg =cfg)
        print("Loaded Pareto Fronts")
        return new_pareto
    
    ## TODO: Generator/mcts.py ~ 973 to 1063


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    # Optional per-run seed for reproducible array jobs.  Existing jobs retain
    # their previous nondeterministic behavior when the variable is unset.
    random_seed = os.environ.get("GARGOYLES_RANDOM_SEED")
    if random_seed is not None:
        import random
        import torch

        random_seed = int(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        print(f"GARGOYLES_RANDOM_SEED={random_seed}")

    sampler = Sampler(cfg, model_dir=cfg["sample"]["model_dir"], model_ver=cfg["sample"]["model_ver"])
    reward_module = getReward(name=cfg["mcts"]["reward_name"])
    if cfg["mcts"]["reward_name"] == "Boltzina":
        reward_module.path_to_workspace = hydra.utils.get_original_cwd()+cfg["mcts"]["work_space"]

    install_request_timeout()
    reward_module = BudgetedReward(reward_module,
        journal=os.environ.get("GOBLET_REWARD_JOURNAL"))
    smiles = cfg["mcts"]["init_smiles"]
    mcts = CoarseMCTS(seed_smiles=smiles, reward_module=reward_module, sampler=sampler)
    # Prepare output paths before search so checkpoints retain partial results.
    output_path = Path(hydra.utils.get_original_cwd()) / "data/result" / cfg["mcts"]["output_fname"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2)

    def save_progress(search):
        frame = search.generated_smiles.copy()
        for column in ("SMILES", "Reward", "Step", "Imp", "FineStepComplete"):
            if column not in frame:
                frame[column] = pd.Series(dtype="object")
        frame["Sim"] = [DataStructs.TanimotoSimilarity(seed_fp,
            AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 2)) for s in frame["SMILES"]]
        frame["Imp"] = frame["Reward"] - search.init_score
        frame = frame.sort_values("Reward", ascending=False)
        temporary = output_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output_path)
        status = {"requested_steps": int(cfg["mcts"]["n_step"]),
            "completed_steps": search.completed_steps, "stop_reason": search.stop_reason,
            "complete": search.completed_steps >= int(cfg["mcts"]["n_step"]),
            "rows": len(frame), "max_reward": search.max_score,
            "initial_reward": search.init_score, "reward_calls": reward_module.calls,
            "cache_hits": reward_module.cache_hits,
            "unique_successful_rewards": len(reward_module.cache),
            "coarse_updates": search.coarse_updates, "saved_at_epoch": time.time()}
        status_path = output_path.with_suffix(".status.json")
        status_tmp = status_path.with_suffix(".json.tmp")
        status_tmp.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        os.replace(status_tmp, status_path)

    mcts.checkpoint = save_progress
    try:
        mcts.search(n_step_coarse=cfg["mcts"]["n_step"],
                    n_step_fine=cfg["mcts"]["n_step_fine"])
    except Exception:
        mcts.stop_reason = "error"
        save_progress(mcts)
        raise
    print(f"Saved {output_path}; completed_steps={mcts.completed_steps}; reason={mcts.stop_reason}", flush=True)


if __name__ == '__main__':
    main()
