import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Utils.reward import getReward
from Utils.utils import read_smilesset
from Utils.sascore import calculateScore

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')

REWARD_NAME = "QED"


def make_csv():
    smiles_list = read_smilesset(hydra.utils.get_original_cwd() + "/data/zinc_250k.smi")
    plogp_list = []
    reward_module = getReward(name=REWARD_NAME)

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        plogp = reward_module.reward(mol)
        plogp_list.append(plogp)

    df = pd.DataFrame()
    df["SMILES"] = smiles_list
    df[REWARD_NAME] = plogp_list
    df = df.sort_values(REWARD_NAME, ascending=True)
    df.to_csv(hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-min800.csv",
              index=False)


def make_range():
    df = pd.read_csv(hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-min800.csv")
    df = df[(df["QED"] >= 0.6) & (df["QED"] < 0.7)]

    smiles_list = df["SMILES"].to_list()
    fp_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 2)
        fp_list.append(fp)

    filtered_list = []
    counter = 0
    while len(filtered_list) < 100:
        counter += 1
        n = np.random.choice(list(range(len(smiles_list))))

        if len(filtered_list) == 0:
            filtered_list.append(n)
        else:
            flag = True
            for i in filtered_list:
                sim = DataStructs.TanimotoSimilarity(fp_list[n], fp_list[i])
                # if sim > 0.14390390520319118:
                #     flag = False

            if flag:
                filtered_list.append(n)

        if counter > 100000:
            break

    df = df.iloc[filtered_list]
    df.to_csv(hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv",
              index=False)

    for smiles in df["SMILES"].to_list():
        print(smiles)


def calc_sa():
    for n in tqdm(range(100)):
        df = pd.read_csv(
            hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/result-mermaid/{n}.csv")

        sa_list = []
        for smiles in df["SMILES"].to_list():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                sa = calculateScore(mol)
            else:
                sa = pd.NA
            sa_list.append(sa)
        df["SA"] = sa_list
        df.to_csv(hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/result-mermaid/result{n}.csv",
                  index=False)


def eda_sa():
    avg_sim_list = []

    for n in tqdm(range(100)):
        # df = pd.read_csv(
        #     hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/result/result{n}.csv")
        df = pd.read_csv(
            hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/result-mermaid/result{n}.csv")
        df = df.dropna()

        avg_sim_list.append(np.mean(df["SA"].to_list()))

    print(np.mean(avg_sim_list), np.std(avg_sim_list))


def eda_sim():
    """
    result: mean 0.14390390520319118, std 0.06335191173125158
    """
    # df = pd.read_csv(
    #     hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-min800.csv")
    # df = df[(df["QED"] >= 0.6) & (df["QED"] < 0.7)]

    avg_sim_list = []

    for n in tqdm(range(100)):
        df = pd.read_csv(
            hydra.utils.get_original_cwd() + f"/Experiment/{REWARD_NAME}/result/result{n}.csv")

        smiles_list = df["SMILES"].to_list()
        fp_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprint(mol, 2)
            fp_list.append(fp)

        sim_list = []
        for i in range(len(smiles_list)-1):
            for j in range(i+1, len(smiles_list)):
                sim = DataStructs.TanimotoSimilarity(fp_list[i], fp_list[j])
                sim_list.append(sim)

        # print(np.mean(sim_list), np.std(sim_list))
        avg_sim_list.append(np.mean(sim_list))

    print(np.mean(avg_sim_list), np.std(avg_sim_list))


def tsne(cfg, n, n_sample=100):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")

    gent_graph_df = pd.read_csv(hydra.utils.get_original_cwd()
                                + f"/Experiment/{REWARD_NAME}/result/result{n}.csv")
    gent_smiles_df = pd.read_csv(hydra.utils.get_original_cwd()
                                 + f"/Experiment/{REWARD_NAME}/result-mermaid/{n}.csv")
    # gent_graph_df = gent_graph_df[gent_graph_df["Imp"] > 0]
    # gent_smiles_df = gent_smiles_df[gent_smiles_df["Imp"] > 0]
    gent_graph_df = gent_graph_df.sample(n_sample)
    gent_smiles_df = gent_smiles_df.sample(n_sample)

    seed_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(seed_df["SMILES"][n]), 2)
    seed_fp_np = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(seed_fp, seed_fp_np)
    seed_fp_list = list(seed_fp_np)

    fp_graph_list = []
    fp_mermaid_list = []

    for smiles in gent_graph_df["SMILES"].to_list():
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        ar = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, ar)
        fp_graph_list.append(ar)

    for smiles in gent_smiles_df["SMILES"].to_list():
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        ar = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, ar)
        fp_mermaid_list.append(list(ar))

    x = np.concatenate([np.array(fp_graph_list), np.array(fp_mermaid_list), np.array([seed_fp_list])], axis=0)
    labels = np.concatenate([np.array([1 for _ in range(len(fp_graph_list))]), np.array([2 for _ in range(len(fp_mermaid_list))]), np.array([0])], axis=0)
    print(x.shape, labels.shape)


    latent_vecs = TSNE(n_components=2).fit_transform(x)
    df = pd.DataFrame(data={'x': latent_vecs[:, 0],
                            'y': latent_vecs[:, 1],
                            'label': labels})

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette=sns.color_palette('hls', 3))
    sns.despine()
    ax.set(xlabel="dim 1", ylabel="dim 2")
    plt.show()


def plot_step(n):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")
    gent_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/result-10000/result{n}.csv")

    print(gent_df.corr())

    max_qeds = []
    t = 0
    width = 10000
    for i in tqdm(range(width*t, width*(t+1))):
        tmp_df = gent_df[(gent_df["Step"] <= i+50) & (gent_df["Step"] > i)]
        if len(tmp_df) > 0:
            # r = np.mean(tmp_df["Sim"].to_list())
            r = tmp_df["Reward"].mean()
            max_qeds.append(r)
        else:
            max_qeds.append(seed_df["QED"][n])
            # max_qeds.append(-1000)

    for i in range(len(max_qeds)):
        if max_qeds[i] == -1000:
            if i == 0:
                max_qeds[i] = 0
            else:
                max_qeds[i] = max_qeds[i-1]

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(list(range(width)), max_qeds)
    sns.despine()
    ax.set(xlabel="Step", ylabel="QED",
           ylim=(0, 1)
           )
    plt.show()


def tsne_step(n, n_sample=100):
    seed_df = pd.read_csv(hydra.utils.get_original_cwd()
                          + f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-range0607.csv")

    gent_graph_df = pd.read_csv(hydra.utils.get_original_cwd()
                                + f"/Experiment/{REWARD_NAME}/result-10000/result{n}.csv")
    # gent_graph_df = gent_graph_df[gent_graph_df["Imp"] > 0]
    # gent_smiles_df = gent_smiles_df[gent_smiles_df["Imp"] > 0]
    # gent_graph_df = gent_graph_df.sample(n_sample)

    seed_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(seed_df["SMILES"][n]), 2)
    seed_fp_np = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(seed_fp, seed_fp_np)
    seed_fp_list = list(seed_fp_np)

    fp_graph_list = []
    label_list = []

    for i, smiles in enumerate(gent_graph_df["SMILES"].to_list()):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        ar = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, ar)
        fp_graph_list.append(ar)
        label_list.append(int((gent_graph_df["Step"][i]-1)/100))

    label_list.append(100)
    print(len(set(label_list)))
    x = np.concatenate([np.array(fp_graph_list), np.array([seed_fp_list])], axis=0)
    # x = np.array(fp_graph_list)
    labels = np.array(label_list)
    print(x.shape, labels.shape)

    latent_vecs = TSNE(n_components=2).fit_transform(x)
    df = pd.DataFrame(data={'x': latent_vecs[:, 0],
                            'y': latent_vecs[:, 1],
                            'label': labels})

    norm = plt.Normalize(df['label'].min(), df['label'].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette="RdBu", s=5)
    sns.despine()
    ax.set(xlabel="dim 1", ylabel="dim 2")
    ax.axvline(x=df["x"][0], linewidth=1, color="g")
    ax.axhline(y=df["y"][0], linewidth=1, color="g")
    ax.get_legend().remove()
    ax.figure.colorbar(sm)
    plt.show()


def correlation(n):
    gent_graph_df = pd.read_csv(hydra.utils.get_original_cwd()
                                + f"/Experiment/{REWARD_NAME}/result-10000/result{n}.csv")

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.scatterplot(data=gent_graph_df, x='Step', y='Sim', palette="RdBu", s=5)
    sns.despine()
    ax.set(xlabel="Step", ylabel="Sim",
           # ylim=(0, 1)
           )
    plt.show()


@hydra.main(config_path="../../config/", config_name="config")
def main(cfg: DictConfig):
    plot_step(0)


if __name__ == '__main__':
    main()

