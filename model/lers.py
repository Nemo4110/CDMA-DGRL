import sys; sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GINEConv, GENConv, GATConv
from torch_geometric.nn import to_hetero
from typing import List, Tuple

from dataset.hgs import DiscreteTimeHeteroGraph
from utils.config import HeteroGraphConfig, MappingManager
from model.layers import LinksPredictor, PositionalEncoding


class SingelGnn(nn.Module):
    r"""Chosen from this cheatsheet: <https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html>"""
    def __init__(self, hidden_dim, gnn_type, gnn_layer_num: int = 2):
        super().__init__()
        assert gnn_layer_num > 0
        self.hidden_dim = hidden_dim

        if gnn_type == "GINEConv":
            self.layers = nn.ModuleList([GINEConv(nn=nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
                                         for _ in range(gnn_layer_num)])
        elif gnn_type == "GENConv":
            self.layers = nn.ModuleList([GENConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, msg_norm=True)
                                         for _ in range(gnn_layer_num)])
        elif gnn_type == "GATConv":
            self.layers = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, add_self_loops=False, edge_dim=self.hidden_dim)
                                         for _ in range(gnn_layer_num)])
        else:
            raise f"Do not support arg:gnn_type={gnn_type} now!"

    def forward(self, node_feats, edge_index, edge_attrs):
        for conv in self.layers:
            node_feats = conv(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        return node_feats


class MultiGnns(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 max_timestep: int,
                 gnn_type: str,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 gnn_layer_num: int = 2,
                 ):
        super().__init__()

        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
        # Solution: use the `nn.ModuleList` instead of list
        self.gnns = nn.ModuleList([SingelGnn(hidden_dim=hidden_dim, gnn_type=gnn_type, gnn_layer_num=gnn_layer_num)
                                   for _ in range(max_timestep)])  # as many fc as max_timestep

        # !!! Warning: here must use `self.gnns[i]`, if we use `gnn` will cause failure of `to_hetero`,
        #              because the `gnn` are temp parameter!
        for i, gnn in enumerate(self.gnns):
            self.gnns[i] = to_hetero(gnn, metadata=(node_types, edge_types))

    def forward(self, list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs):
        for i in range(len(self.gnns)):
            list_dict_node_feats[i] = self.gnns[i](list_dict_node_feats[i],
                                                   list_edge_index_dict[i],
                                                   list_dict_edge_attrs[i])

        return list_dict_node_feats


class LinksPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.re_weight_a = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.re_weight_b = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, node_features_a, node_features_b, edge_label_index):
        node_features_a_selected = node_features_a[edge_label_index[0]]
        node_features_b_selected = node_features_b[edge_label_index[1]]

        node_features_a_selected = self.re_weight_a(node_features_a_selected)
        node_features_b_selected = self.re_weight_b(node_features_b_selected)

        return (node_features_a_selected * node_features_b_selected).sum(dim=-1)


class LERS(nn.Module):
    def __init__(self,
                 max_timestep: int,
                 gnn_type: str,
                 neg_smp_strategy: int,

                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],

                 num_decoder_layers=6,
                 hidden_dim: int = 128,
                 gnn_layer_num: int = 2):
        super().__init__()

        self.max_timestep = max_timestep
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num
        self.neg_smp_strategy = neg_smp_strategy

        self.node_types = node_types
        self.edge_types = edge_types

        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim

        # EMBD
        # ~~Think about the first args of `nn.Embedding` here should equal to~~
        #  - max of total `HADM_ID`?
        #  - current batch_size of `HADM_ID`?
        #  √ Or just deprecate the `nn.Embedding` as we already have the node_features
        self.module_dict_embedding = nn.ModuleDict({
            node_type: nn.Embedding(MappingManager.node_type_to_node_num[node_type], self.hidden_dim) \
            for node_type in self.node_types if node_type != 'admission'
        })

        # PROJ
        self.module_dict_node_feat_proj = nn.ModuleDict({
            node_type: nn.Linear(in_features=MappingManager.node_type_to_node_feat_dim_in[node_type], out_features=self.hidden_dim) \
            for node_type in self.node_types
        })
        self.module_dict_edge_feat_proj = nn.ModuleDict({
            # TypeError: module name should be a string. Got tuple
            "_".join(edge_type): nn.Linear(in_features=MappingManager.edge_type_to_edge_feat_dim_in[edge_type], out_features=self.hidden_dim) \
            for edge_type in self.edge_types
        })

        self.position_encoding = PositionalEncoding(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep)

        self.gnns = MultiGnns(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep,
                              gnn_type=self.gnn_type, gnn_layer_num=self.gnn_layer_num,
                              node_types=self.node_types, edge_types=self.edge_types)

        # DECODER
        self.module_dict_decoder = nn.ModuleDict({
            node_type: nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512),
                num_layers=self.num_decoder_layers
            ) for node_type in self.node_types
        })

        # LIKES_PREDICTOR
        self.module_dict_links_predictor = nn.ModuleDict({
            node_type: LinksPredictor(hidden_dim=self.hidden_dim) \
            for node_type in self.node_types if node_type != "admission"
        })

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, hg: HeteroData):
        hgs = [MyOwnDataset.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=self.neg_smp_strategy) for t in range(self.max_timestep)]

        list_dict_node_feats = [{
           node_type: self.module_dict_node_feat_proj[node_type](hg[node_type].x) + self.module_dict_embedding[node_type](hg[node_type].node_id) if node_type != 'admission' \
                 else self.module_dict_node_feat_proj[node_type](hg[node_type].x) \
            for node_type in self.node_types
        } for hg in hgs]
        list_edge_index_dict = [{
            edge_type: edge_index \
            for edge_type, edge_index in hg.edge_index_dict.items() if edge_type in self.edge_types
        } for hg in hgs]
        list_dict_edge_attrs = [{
            edge_type: self.module_dict_edge_feat_proj["_".join(edge_type)](hg[edge_type].x)
            for edge_type in self.edge_types
        } for hg in hgs]

        # go through gnns
        list_dict_node_feats = self.gnns(list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs)

        # decode
        dict_node_feat = {
            node_type: torch.stack([dict_node_feats[node_type] for dict_node_feats in list_dict_node_feats]) \
            for node_type in self.node_types
        }
        for node_type, node_feat in dict_node_feat.items():
            node_feat = self.position_encoding(node_feat)  # Add position encoding
            tgt_mask = memory_mask = nn.Transformer.generate_square_subsequent_mask(self.max_timestep).to(node_feat.device)
            node_feat = self.module_dict_decoder[node_type](tgt=node_feat, memory=node_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dict_node_feat[node_type] = node_feat  # update

        # Link predicting:
        dict_every_day_pred = {}
        for node_type in self.node_types:
            if node_type != "admission":  # admission is not the goal of prediction
                if node_type == "drug":
                    dict_every_day_pred[node_type] = {
                        "scores": [self.module_dict_links_predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                                               dict_node_feat["drug"][curr_timestep],
                                                                               hg.labels4drug_index) \
                                   for curr_timestep, hg in enumerate(hgs)],
                        "labels":  [hg.labels4drug       for hg in hgs],
                        "indices": [hg.labels4drug_index for hg in hgs]
                    }
                elif node_type == "labitem":
                    dict_every_day_pred[node_type] = {
                        "scores": [self.module_dict_links_predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                                               dict_node_feat["labitem"][curr_timestep],
                                                                               hg.lables4item_index) \
                                   for curr_timestep, hg in enumerate(hgs)],
                        "labels":  [hg.lables4item       for hg in hgs],
                        "indices": [hg.lables4item_index for hg in hgs]
                    }
                else:
                    raise f"check the node_types config! curr: {self.node_types}"

        return dict_every_day_pred


if __name__ == "__main__":
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    model = LERS(max_timestep=20,
                 gnn_type="GENConv",
                 node_types=node_types,
                 edge_types=edge_types,
                 num_decoder_layers=6,
                 hidden_dim=128,
                 neg_smp_strategy=0)
    print(model)
