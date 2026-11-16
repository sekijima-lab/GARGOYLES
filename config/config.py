from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Tuple
import math
from Utils.mol_utils import ATOM_IDX, BOND_IDX


@dataclass
class EmbConfig:
    infeat_atom: int = len(ATOM_IDX)
    infeat_bond: int = len(BOND_IDX)
    hidden: int = 128
    outfeat: int = 128


@dataclass
class NodeConfig:
    in_channels: int = 128
    hidden_channels: int = 64
    out_channels: int = len(ATOM_IDX)+1


@dataclass
class EdgeConfig:
    num_embeddings: int = len(ATOM_IDX)+1
    embedding_dim: int = 64
    h_graph_dim: int = 128
    input_size_rnn: int = 128
    hidden_size_rnn: int = 256
    num_layers: int = 2
    hidden_size_head: int = 64
    out_size_head: int = len(BOND_IDX)+1


@dataclass
class TrainConfig:
    data_size: int = 1000000
    batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = True
    lr: float = 0.0001
    start_epoch: int = 0
    epoch: int = 100
    loss_weight: Tuple[int] = (1, 1)
    eval_step: int = 1
    save_step: int = 5
    model_dir: str = "/ckpt/pretrain/ver2/"
    log_dir: str = "/log/"
    log_filename: str = "pretrain-ver2.txt"
    # vocab: str = "/data/vocabulary/vocab-ring.pickle"
    bond_loss_weight: Tuple[float] = (1., 20., 20., 20.)

@dataclass
class SamplingConfig:
    model_dir: str = "/ckpt/pretrain/ver2/"
    max_step: int = 20
    num: int = 10000
    model_ver: int = 20
    seed_smiles: str = "C"


@dataclass
class ExperimentConfig:
    n_iter: int = 20
    seed_file: str = "/Experiment/QED/seed_lowest_QED.csv"
    out_dir: str = "/Experiment/QED/result-10000/"
    model_dir: str = "/ckpt/pretrain/ver2/"
    model_ver: int = 20

@dataclass
class MCTSConfig:
    n_step: int = 1000
    n_step_fine: int = 20
    n_iter: int = 3
    seq_len: int = 25
    in_smiles_file: str = "/Data/input/init_smiles.smi" #TODO: merge to data_dir
    rep_file: str = ""
    modeL_dir: str = "/ckpt/"
    out_dir: str = "/Data/output/" # TODO: merge to data_dir
    ucb_c: float = 1/math.sqrt(2)
    model_ver: int = 100
    model_dir: str = "/ckpt/"
    reward_name: str = "PLogP"
    data_dir: str = "/data_templete/"
    isLoadTree: bool = False
    time_limit_sec: int = 10*60
    sascore_threshold: float = 3.5
    tanimoto_threshold: float = 0.0
    output_fname: str = "output.csv"
    work_space: str = "/wkdir/0/"
    init_smiles: str = "c1ccccc1"
    


@dataclass
class Config:
    emb: EmbConfig = EmbConfig()
    node: NodeConfig = NodeConfig()
    edge: EdgeConfig = EdgeConfig()
    train: TrainConfig = TrainConfig()
    sample: SamplingConfig = SamplingConfig()
    # rl: RLConfig = RLConfig()
    exp: ExperimentConfig = ExperimentConfig()
    mcts: MCTSConfig = MCTSConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
