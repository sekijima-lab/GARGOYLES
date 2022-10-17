import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Utils.reward import getReward
from Generator.sampling import Sampler
from Generator.mcts import CoarseMCTS, FineMCTS
from Utils.utils import read_smilesset

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf
import seaborn as sns
import matplotlib.pyplot as plt

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')

REWARD_NAME = "QED"
OUT_DIR = "/Experiment/QED/result-ver2/"


def search(seed_smiles, n_iter, sampler, out_dir, out_filename, th):
    reward_module = getReward(name=REWARD_NAME, seed_smiles=seed_smiles, th=th)
    max_score = reward_module.vmin

    mcts = CoarseMCTS(seed_smiles=seed_smiles, reward_name=REWARD_NAME, sampler=sampler)
    # mcts = FineMCTS(seed_smiles=seed_smiles, reward_module=reward_module, sampler=sampler)
    # mcts.search(n_step=1000)
    mcts.search(n_step_coarse=20, n_step_fine=100)

    df = pd.DataFrame()

    sim_list = []
    seed_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(seed_smiles), 2)
    for gent_smiles in mcts.generated_smiles["SMILES"]:
        gent_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(gent_smiles), 2)
        sim = DataStructs.TanimotoSimilarity(seed_fp, gent_fp)
        sim_list.append(sim)

    df["SMILES"] = mcts.generated_smiles["SMILES"]
    df["Reward"] = mcts.generated_smiles["Reward"]
    df["Sim"] = sim_list
    df["Step"] = mcts.generated_smiles["Step"]
    df["Imp"] = df["Reward"] - reward_module.reward(Chem.MolFromSmiles(seed_smiles))
    df = df.sort_values("Reward", ascending=False)
    df.to_csv(hydra.utils.get_original_cwd() + out_dir + out_filename, index=False)


def run_exp(cfg):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    seed_list = seed_df["SMILES"].to_list()[:800]

    for i, smiles in enumerate(seed_list):

        search(seed_smiles=smiles, n_iter=cfg["exp"]["n_iter"],
               sampler=Sampler(cfg, model_dir="/ckpt/pretrain/ver2/", model_ver=cfg["sample"]["model_ver"]),
               out_dir=OUT_DIR,
               out_filename=f"result{i}.csv",
               th=0.6,
               )
        # try:
        #     search(seed_smiles=smiles, n_iter=cfg["exp"]["n_iter"], sampler=Sampler(cfg), out_dir=OUT_DIR,
        #            out_filename=f"result{i}.csv", th=0.6)
        # except:
        #     pass


def evaluate(cfg, n=200):
    imp_list = []
    qed1_list = []
    qed2_list = []
    qed3_list = []
    qed50_list = []
    avg_qed_list = []
    sim_list = []
    nov_list = []
    uni_list = []
    sc_list = []
    seed_df = pd.read_csv(hydra.utils.get_original_cwd() + f"/zinc-{REWARD_NAME.lower()}-range0607.csv")
    seed_qed = seed_df[REWARD_NAME].to_list()

    for i in tqdm(range(n)):
        try:
            # gent_df = pd.read_csv(hydra.utils.get_original_cwd() + f"/result/result{i}.csv")
            gent_df = pd.read_csv(hydra.utils.get_original_cwd() + f"/result-mermaid/{i}.csv")
            if len(gent_df) == 0:
                qed1_list.append(seed_df[REWARD_NAME][i])
                qed2_list.append((seed_df[REWARD_NAME][i]))
                qed3_list.append((seed_df[REWARD_NAME][i]))
                qed50_list.append((seed_df[REWARD_NAME][i]))
                avg_qed_list.append((seed_df[REWARD_NAME][i]))
                imp_list.append(0)
                sim_list.append(0)
                sc_list.append(0)
            else:
                qed1_list.append(max(gent_df["Reward"][0], seed_qed[i]))
                qed2_list.append(max(gent_df["Reward"][1], seed_qed[i]))
                qed3_list.append(max(gent_df["Reward"][2], seed_qed[i]))
                qed50_list.append(max(gent_df["Reward"][49], seed_qed[i]))
                avg_qed_list.append(max(np.mean(gent_df["Reward"][:49]), seed_qed[i]))
                imp_list.append(max(gent_df["Reward"][0] - seed_qed[i], 0))
                sim_list.append(gent_df["Sim"][0])
                if imp_list[-1] > 0:
                    sc_list.append(1)

        except FileNotFoundError:
            print(i)
            pass

    print(f"Success rate is {float(np.mean(sc_list))}")
    print(f"Average top1 {REWARD_NAME} is {float(np.mean(qed1_list))} and STD is {float(np.std(qed1_list))}")
    print(f"Average top2 {REWARD_NAME} is {float(np.mean(qed2_list))} and STD is {float(np.std(qed2_list))}")
    print(f"Average top3 {REWARD_NAME} is {float(np.mean(qed3_list))} and STD is {float(np.std(qed3_list))}")
    print(f"Average top50 {REWARD_NAME} is {float(np.mean(qed50_list))} and STD is {float(np.std(qed50_list))}")
    print(f"Average avg top50 {REWARD_NAME} is {float(np.mean(avg_qed_list))} and STD is {float(np.std(avg_qed_list))}")
    print(f"Average Similarity is {float(np.mean(sim_list))} and STD is {float(np.std(sim_list))}")
    print(f"Average {REWARD_NAME} improvement is {float(np.mean(imp_list))} and STD is {float(np.std(imp_list))}")
    print(seed_df.mean(), seed_df.std())

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.distplot(seed_qed, bins=[0.6+i*0.02 for i in range(20)])
    sns.distplot(qed1_list, bins=[0.6+i*0.02 for i in range(20)])
    sns.despine()
    ax.set(xlabel="QED", ylabel="Number of Molecules",
           ylim=(0, 80)
           )
    plt.show()


def calc_prop(cfg, n=200):
    nov_list = []
    uni_list = []
    sc_list = []
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    seed_qed = seed_df[REWARD_NAME].to_list()
    zinc_list = read_smilesset(hydra.utils.get_original_cwd() + "/data/zinc_250k.smi")
    zinc_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False) for smiles in tqdm(zinc_list)]

    for i in tqdm(range(n)):
        try:
            gent_df = pd.read_csv(hydra.utils.get_original_cwd()
                                  + f"/Experiment/{REWARD_NAME}/result/result{i}.csv")
            gent_df = gent_df[gent_df["Reward"] > seed_qed[i]]
            gent_list = gent_df["SMILES"].to_list()
            num_nov = 0
            for smiles in gent_list:
                if smiles not in zinc_list:
                    num_nov += 1
            nov_list.append(num_nov/len(gent_list))
            uni_list.append(len(set(gent_list))/len(gent_list))

        except FileNotFoundError:
            print(i)
            pass

    print(f"Success rate is {float(np.mean(sc_list))}")
    print(f"Average Novelty is {float(np.mean(nov_list))} and STD is {float(np.std(nov_list))}")
    print(f"Average Uniqueness is {float(np.mean(uni_list))} and STD is {float(np.std(uni_list))}")


def plot_smiles(n):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    gent_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/result-mermaid/{n}.csv")

    print(seed_df["SMILES"][n])
    for smiles in gent_df["SMILES"][:20]:
        print(smiles)


def plot_step(n):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    gent_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/result-ver2/result{n}.csv")

    max_qeds = []
    t = 0
    width = 2000
    for i in tqdm(range(width*t, width*(t+1))):
        tmp_df = gent_df[(gent_df["Step"] <= i+20) & (gent_df["Step"] > i)]
        if len(tmp_df) > 0:
            r = np.mean(tmp_df["Reward"].to_list())
            # r = max(r, seed_df["QED"][n])
            max_qeds.append(r)
        else:
            max_qeds.append(seed_df["QED"][n])

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(list(range(width)), max_qeds)
    sns.despine()
    ax.set(xlabel="Step", ylabel="QED",
           # ylim=(0, 1)
           )
    plt.show()


@hydra.main(config_path="../../config/", config_name="config")
def main(cfg: DictConfig):
    # run_exp(cfg)
    evaluate(cfg, n=100)
    # calc_prop(cfg, n=100)
    # plot_smiles(10)
    # plot_step(0)


if __name__ == '__main__':
    main()
