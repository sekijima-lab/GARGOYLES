from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Tuple
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
    num: int = 100
    model_ver: int = 20
    seed_smiles: str = "C"


# @dataclass
# class RLConfig:
#     init_smiles: str = "CCOc1ccc(OCC)c([C@H]2C(C#N)=C(N)N(c3ccccc3C(F)(F)F)C3=C2C(=O)CCC3)c1"
#     vocab: str = "/data/vocabulary/vocab-ring.pickle"
#     model_ver_frag: int = 20
#     model_ver_value: int = 0
#     model_dir: str = "/ckpt/pretrain/frag/normal/"
#     max_ep_len: int = 10  # max timesteps in one episode
#     max_training_timesteps: int = int(3e6)  # break training loop if timeteps > max_training_timesteps
#     print_freq: int = max_ep_len * 64  # print avg reward in the interval (in num timesteps)
#     log_freq: int = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
#     save_model_freq: int = 1000  # save model frequency (in num timesteps)
#     action_std: float = 0.6  # starting std for action distribution (Multivariate Normal)
#     update_timestep: int = max_ep_len * 64  # update policy every n timesteps
#     K_epochs: int = 10  # update policy for K epochs in one PPO update
#     eps_clip: float = 0.2  # clip parameter for PPO
#     gamma: float = 0.99  # discount factor
#     lr_actor: float = 0.0003  # learning rate for actor network
#     lr_critic: float = 0.001  # learning rate for critic network
#     random_seed: float = 0  # set random seed if required (0 = no random seed)
#     ckpt: str = "/ckpt/rl/"
#     log_dir: str = "/log/rl/"
#     log_filename: str = "ppo.txt"


@dataclass
class ExperimentConfig:
    n_iter: int = 20
    seed_file: str = "/Experiment/QED/seed_lowest_QED.csv"
    out_dir: str = "/Experiment/QED/result-10000/"
    model_dir: str = "/ckpt/pretrain/ver2/"
    model_ver: int = 20


@dataclass
class Config:
    emb: EmbConfig = EmbConfig()
    node: NodeConfig = NodeConfig()
    edge: EdgeConfig = EdgeConfig()
    train: TrainConfig = TrainConfig()
    sample: SamplingConfig = SamplingConfig()
    # rl: RLConfig = RLConfig()
    exp: ExperimentConfig = ExperimentConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)