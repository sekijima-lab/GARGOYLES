import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Utils.reward import getReward
from Utils.utils import read_smilesset

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')

REWARD_NAME = "PLogP"


@hydra.main(config_path="../../config/", config_name="config")
def main(cfg: DictConfig):
    smiles_list = read_smilesset(hydra.utils.get_original_cwd()+"/data/zinc_250k.smi")
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
    df.to_csv(hydra.utils.get_original_cwd()+f"/Experiment/{REWARD_NAME}/zinc-{REWARD_NAME.lower()}-min800.csv",
              index=False)


if __name__ == '__main__':
    main()
