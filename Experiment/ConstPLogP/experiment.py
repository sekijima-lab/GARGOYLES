import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Utils.reward import getReward
from Generator.sampling import Sampler
from Generator.mcts import CoarseMCTS

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')

REWARD_NAME = "ConstPLogP"
OUT_DIR = "/Experiment/ConstPLogP/result"


def search(seed_smiles, n_iter, sampler, out_dir, out_filename, th):
    reward_module = getReward(name=REWARD_NAME, seed_smiles=seed_smiles, th=th)
    gen_df = pd.DataFrame(columns=["SMILES", "Reward", "Step"])
    max_score = reward_module.vmin

    mcts = CoarseMCTS(seed_smiles=seed_smiles, reward_name=REWARD_NAME, sampler=sampler)
    mcts.search(n_step_coarse=2000, n_step_fine=100)

    gen_df = gen_df.sort_values("Reward", ascending=False)

    sim_list = []
    mol = Chem.MolFromSmiles(seed_smiles)
    seed_fp = AllChem.GetMorganFingerprint(mol, 2)
    for smiles in gen_df["SMILES"].to_list():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            gent_fp = AllChem.GetMorganFingerprint(mol, 2)
            sim = DataStructs.TanimotoSimilarity(seed_fp, gent_fp)
        else:
            sim = 0
        sim_list.append(sim)
    gen_df["Sim"] = sim_list
    gen_df["Imp"] = gen_df["Reward"] - reward_module.reward(Chem.MolFromSmiles(seed_smiles))
    gen_df.to_csv(hydra.utils.get_original_cwd() + out_dir + out_filename, index=False)


def run_exp(cfg):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    seed_list = seed_df["SMILES"].to_list()[:800]

    for i, smiles in enumerate(seed_list):
        try:
            search(seed_smiles=smiles, n_iter=cfg["exp"]["n_iter"], sampler=Sampler(cfg), out_dir=OUT_DIR,
                   out_filename=f"result{i}.csv", th=0.6)
        except:
            pass


def evaluate(cfg, n=200):
    imp_list = []
    qed_list = []
    sim_list = []
    sc_list = []
    seed_df = pd.read_csv(hydra.utils.get_original_cwd() + "/zinc-plogp-min800.csv")
    seed_qed = seed_df["PLogP"].to_list()

    for i in tqdm(range(n)):
        try:
            path = hydra.utils.get_original_cwd() + f"/result/result{i}.csv"
            gent_df = pd.read_csv(hydra.utils.get_original_cwd() + f"/result/result{i}.csv")
            if len(gent_df) == 0:
                qed_list.append(seed_df["PLogP"][i])
                imp_list.append(0)
                sim_list.append(0)
                sc_list.append(0)
            else:
                qed_list.append(max(gent_df["Reward"][0], seed_qed[i]))
                imp_list.append(max(gent_df["Reward"][0] - seed_qed[i], 0))
                sim_list.append(gent_df["Sim"][0])
                if imp_list[-1] > 0:
                    sc_list.append(1)

        except FileNotFoundError:
            print(i)
            pass

    print(f"Success rate is {float(np.mean(sc_list))}")
    print(f"Average {REWARD_NAME} is {float(np.mean(qed_list))} and STD is {float(np.std(qed_list))}")
    print(f"Average Similarity is {float(np.mean(sim_list))} and STD is {float(np.std(sim_list))}")
    print(f"Average {REWARD_NAME} improvement is {float(np.mean(imp_list))} and STD is {float(np.std(imp_list))}")


@hydra.main(config_path="../../config/", config_name="config")
def main(cfg: DictConfig):
    # run_exp(cfg)
    evaluate(cfg, n=800)


if __name__ == '__main__':
    main()
