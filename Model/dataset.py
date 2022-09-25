import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class FragmentDataset(DGLDataset):
    def __init__(self, graph_list, label_list, num_node_list, name="mol"):
        super(FragmentDataset, self).__init__(name=name)
        self.graph_list = graph_list
        self.label_atom_list = []
        self.label_bond_list = []
        self.num_node_list = num_node_list

        print("--- Preprocessing %s Dataset---" % name)
        for label in label_list:
            self.label_atom_list.append(torch.tensor(label["atom"], dtype=torch.long).view(1, -1))
            self.label_bond_list.append(torch.tensor(label["bond"], dtype=torch.long).view(1, -1))
        print("--- Complete ---")

    def __len__(self):
        return len(self.graph_list)

    def process(self):
        pass

    def __getitem__(self, item):
        return self.graph_list[item], self.label_atom_list[item], self.label_bond_list[item], self.num_node_list[item]


def collate_fragment(batch):
    graph_list, label_atom_list, label_bond_list, num_node_list = list(zip(*batch))
    batch_graphs = dgl.batch(graph_list)
    batch_label_atom = torch.cat(label_atom_list, dim=0)
    batch_label_bond = torch.cat(label_bond_list, dim=0)

    return batch_graphs, batch_label_atom, batch_label_bond, num_node_list



def main(cfg):
    with open(hydra.utils.get_original_cwd() + "/data/features/dgl-graph.pickle", mode="rb") as f:
        graph_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + "/data/features/label.pickle", mode="rb") as f:
        label_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + "/data/features/num-node.pickle", mode="rb") as f:
        num_node_list = pickle.load(f)

    sp = int(len(graph_list)*0.8)
    train_set = FragmentDataset(graph_list[:sp], label_list[:sp], num_node_list[:sp], name="Train")
    test_set = FragmentDataset(graph_list[sp:], label_list[sp:], num_node_list[sp:], name="Test")
    train_loader = GraphDataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                                   shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                   collate_fn=collate_fragment)
    test_loader = GraphDataLoader(test_set, batch_size=cfg["train"]["batch_size"],
                                  shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                  collate_fn=collate_fragment)


if __name__ == "__main__":
    with open(hydra.utils.get_original_cwd() + "/data/features/dgl-graph.pickle", mode="rb") as f:
        graph_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + "/data/features/label.pickle", mode="rb") as f:
        label_list = pickle.load(f)
    with open(hydra.utils.get_original_cwd() + "/data/features/num-node.pickle", mode="rb") as f:
        num_node_list = pickle.load(f)

    sp = int(len(graph_list) * 0.8)
    train_set = FragmentDataset(graph_list[:sp], label_list[:sp], num_node_list[:sp], name="Train")
    test_set = FragmentDataset(graph_list[sp:], label_list[sp:], num_node_list[sp:], name="Test")
    train_loader = GraphDataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                                   shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                   collate_fn=collate_fragment)
    test_loader = GraphDataLoader(test_set, batch_size=cfg["train"]["batch_size"],
                                  shuffle=cfg["train"]["shuffle"], drop_last=cfg["train"]["drop_last"],
                                  collate_fn=collate_fragment)
    print("------------")

    for bg, labels in dataloader:
        print(bg.batch_size)
        print(bg.batch_num_nodes())
        feats = bg.ndata['atom']
        print(feats, feats.shape)
        print(bg.nodes().shape)
        print(bg.edata["bond"], bg.edata["bond"].shape)
        print(bg.edges()[0].shape)
        break


