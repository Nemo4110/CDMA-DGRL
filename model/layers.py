import torch
import torch.nn as nn
import torch.nn.functional as fn
import math
import numpy as np

from torch_geometric.nn.conv import GINEConv, GENConv, GATConv
from utils.enum_type import FeatureSource, FeatureType
from model.init import normal_
from typing import List


class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


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


class LinksPredictor(nn.Module):
    def __init__(self, hidden_dim=128, score_func="linear"):
        super().__init__()
        self.re_weight_a = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.re_weight_b = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        self.score_func = score_func
        if self.score_func == "linear":
            self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, cur_day_patient_condition, item_features_selected):
        a = self.re_weight_a(cur_day_patient_condition)
        b = self.re_weight_b(item_features_selected)

        if self.score_func == "linear":
            scores = self.fc(torch.cat([a, b], dim=-1))
            return scores.squeeze(1)
        else:
            return torch.mul(a, b).sum(-1)


def get_decoder_by_choice(choice: str, hidden_dim: int, num_layers: int = 1):
    if choice == "TransformerDecoder":
        chosen_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512),
            num_layers=num_layers
        )
    elif choice == "RNN":
        chosen_decoder = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    elif choice == "GRU":
        chosen_decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    elif choice == "LSTM":
        chosen_decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    else:
        raise NotImplementedError

    return chosen_decoder


def decode(decoder: nn.Module, input_seq: torch.tensor, h_0: torch.tensor):
    """
    Args:
        decoder:
        input_seq:
            has shape (seq_len, n_samples, n_features)
        h_0: for RNN-like modules
            has shape (1, n_samples, n_features)
    """
    if isinstance(decoder, nn.TransformerDecoder):
        tgt_mask = memory_mask = nn.Transformer\
            .generate_square_subsequent_mask(input_seq.shape[0])\
            .to(input_seq.device)
        output_seq = decoder(tgt=input_seq, memory=input_seq, tgt_mask=tgt_mask, memory_mask=memory_mask)
    elif isinstance(decoder, nn.RNN):
        # TypeError: RNN.forward() got an unexpected keyword argument 'h_0'
        output_seq, h_n = decoder(input=input_seq, hx=h_0)
    elif isinstance(decoder, nn.GRU):
        output_seq, h_n = decoder(input=input_seq, hx=h_0)
    elif isinstance(decoder, nn.LSTM):
        h_0, c_0 = h_0, h_0
        output_seq, h_n = decoder(input=input_seq, hx=(h_0, c_0))
    else:
        raise NotImplementedError

    return output_seq


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, pred, label, adm_lens):
        # pred and label have the same shape
        weights = torch.zeros_like(label)
        weights = MaskedBCEWithLogitsLoss.mask_pad_adm(weights, adm_lens)
        self.reduction = 'none'
        unweighted_loss = super(MaskedBCEWithLogitsLoss, self).forward(pred, label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

    @staticmethod
    def mask_pad_adm(X, adm_lens):
        """屏蔽batch中填充的住院天
        X: (B, max_adm_len, med_vocab_size)
        adm_lens: (B,)
        """
        B, max_adm_len = X.size(0), X.size(1)
        for i in range(B):
            X[i, :adm_lens[i], :] = 1.

        return X


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self,
        mask_mat,
        att_hidden_size=(80, 40),
        activation="sigmoid",
        softmax_stag=False,
        return_seq_weight=True,
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(
            self.att_hidden_size, activation=self.activation, bn=False
        )
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T

        queries = queries.repeat(1, hist_len)
        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat(
            [queries, keys, queries - keys, queries * keys], dim=-1
        )
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = mask >= keys_length.unsqueeze(1)

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size**0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output


class FMEmbedding(nn.Module):
    r"""Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 2D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BaseEmbeddingLayer(nn.Module):
    def __init__(self, config, dataset):
        super(BaseEmbeddingLayer, self).__init__()
        self.device = config["device"]
        self.embedding_size = config["embedding_size"]
        self.LABEL = config.get("LABEL_FIELD", "label")

        # 子类初始化完需要调用以下函数！
        # self._get_fields_names_dims(dataset)
        # self._get_embedding_tables()

    def _get_fields_names_dims(self, dataset):
        """分用户user、物品item，获取token和float类型的特征列数量及相应维度"""
        # user
        self.user_token_field_dims = []
        self.user_token_field_names = []
        self.user_float_field_names = []
        for field_name in self.user_field_names:
            if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                self.user_token_field_names.append(field_name)
                self.user_token_field_dims.append(dataset.num(field_name))
            elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                self.user_float_field_names.append(field_name)
            else:
                raise NotImplementedError

        # item
        self.item_token_field_dims = []
        self.item_token_field_names = []
        self.item_float_field_names = []
        for field_name in self.item_field_names:
            if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                self.item_token_field_names.append(field_name)
                self.item_token_field_dims.append(dataset.num(field_name))
            elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                self.item_float_field_names.append(field_name)
            else:
                raise NotImplementedError

    def _get_embedding_tables(self):
        """分用户user、物品item，获取token和float类型的embedding表格"""
        # user
        if len(self.user_token_field_dims) > 0:
            self.user_token_field_offsets = np.array(
                (0, *np.cumsum(self.user_token_field_dims)[:-1]), dtype=np.long)
            self.user_token_embedding_table = FMEmbedding(
                self.user_token_field_dims,
                self.user_token_field_offsets,
                self.embedding_size)
        if len(self.user_float_field_names) > 0:
            self.user_float_embedding_table = nn.Linear(len(self.user_float_field_names), self.embedding_size)

        # item
        if len(self.item_token_field_dims) > 0:
            self.item_token_field_offsets = np.array(
                (0, *np.cumsum(self.item_token_field_dims)[:-1]), dtype=np.long)
            self.item_token_embedding_table = FMEmbedding(
                self.item_token_field_dims,
                self.item_token_field_offsets,
                self.embedding_size)
        if len(self.item_float_field_names) > 0:
            self.item_float_embedding_table = nn.Linear(len(self.item_float_field_names), self.embedding_size)

    def _embed_item_feat_fields(self, item_id):
        original_item_features = self.item_features[item_id]

        # float fields first，要求物品对应特征源df中，float类型的特征列放最前面，目前药物、检验项目的df都满足要求
        if len(self.item_float_field_names) > 0:
            item_float_features = original_item_features[:, :len(self.item_float_field_names)].float()
            item_dense_embeddings = self.item_float_embedding_table(item_float_features).unsqueeze(1)
        else:
            item_dense_embeddings = None

        if len(self.item_token_field_names) > 0:
            item_token_features = original_item_features[:, -len(self.item_token_field_names):].long()
            item_sparse_embeddings = self.item_token_embedding_table(item_token_features)
        else:
            item_sparse_embeddings = None

        if item_dense_embeddings is not None and item_sparse_embeddings is not None:
            return torch.cat([item_dense_embeddings, item_sparse_embeddings], dim=1)
        else:
            return item_sparse_embeddings if item_sparse_embeddings is not None else item_dense_embeddings

    def _embed_user_feat_fields(self, user_id):
        # STEP 1: 先从原始特征数据表中，按user_id取出相应行
        original_user_features = self.user_features[user_id]

        # STEP 2: 处理float类型特征
        if len(self.user_float_field_names) > 0:
            # 要求：user特征表中，float类型的列放最前面
            #      mimic-iii的admission表一开始就满足要求，因为它没有float类型的特征列
            user_float_features = original_user_features[:, :len(self.user_float_field_names)]
            user_dense_embeddings = self.user_float_embedding_table(
                user_float_features).unsqueeze(1)  # [B, n_float_field] -> [B, 1, embedding_size]
        else:
            user_dense_embeddings = None

        # STEP 3：处理token类型特征
        if len(self.user_token_field_names) > 0:
            user_token_features = original_user_features[:, -len(self.user_token_field_names):]
            user_sparse_embeddings = self.user_token_embedding_table(
                user_token_features)
        else:
            user_sparse_embeddings = None

        # STEP 4：合并dense、sparse embedding
        if user_dense_embeddings is not None and user_sparse_embeddings is not None:
            return torch.cat([user_dense_embeddings, user_sparse_embeddings], dim=1).flatten(start_dim=1)
        else:
            return user_sparse_embeddings.flatten(start_dim=1) if user_sparse_embeddings is not None \
                else user_dense_embeddings.flatten(start_dim=1)


class GeneralEmbeddingLayer(BaseEmbeddingLayer):
    def __init__(self, config, dataset):
        super(GeneralEmbeddingLayer, self).__init__(config, dataset)

        self.user_field_names = dataset.fields(source=[FeatureSource.USER, ])
        self.item_field_names = dataset.fields(source=[FeatureSource.ITEM, ])

        self.USER_ID = config.get("USER_ID_FIELD", "user_id")
        self.ITEM_ID = config.get("ITEM_ID_FIELD", "item_id")

        self.n_items = dataset.num_items
        self.item_id_embedding_table = nn.Embedding(self.n_items, self.embedding_size)

        self.user_features = dataset.user_feat_values.to(self.device)
        self.item_features = dataset.item_feat_values.to(self.device)

        self._get_fields_names_dims(dataset)
        self._get_embedding_tables()

    def forward(self, interaction):
        users = torch.from_numpy(interaction[self.USER_ID].values).long().to(self.device)
        items = torch.from_numpy(interaction[self.ITEM_ID].values).long().to(self.device)

        users_embedding = self._embed_user_feat_fields(users)
        items_embedding = self._embed_item_feat_fields(items)

        return users_embedding, items_embedding


class ContextEmbeddingLayer(BaseEmbeddingLayer):
    def __init__(self, config, dataset):
        super(ContextEmbeddingLayer, self).__init__(config, dataset)

        self.field_names = dataset.fields(source=[FeatureSource.USER, FeatureSource.ITEM_ID, FeatureSource.ITEM])

        self.double_tower = config.get("double_tower", None)
        if self.double_tower is None:
            self.double_tower = False

        if self.double_tower:  # 也就是说分物品塔和用户塔
            # user_id不应该作为用户塔的特征，
            # 否则按目前以adm划分训练、测试、验证的做法，在后面两个集合的user_id对应的embbeding向量不会被优化到
            self.user_field_names = dataset.fields(source=[FeatureSource.USER,])
            self.item_field_names = dataset.fields(source=[FeatureSource.ITEM_ID, FeatureSource.ITEM])
            self.field_names = self.user_field_names + self.item_field_names  # 先用户，后物品

            # 统计各类型的特征列数量
            # 用户塔
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            for field_name in self.user_field_names:
                if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                    self.user_float_field_num += 1
                else:
                    raise NotImplementedError
            # 物品塔
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            for field_name in self.item_field_names:
                if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                    self.item_float_field_num += 1
                else:
                    raise NotImplementedError

        self._get_fields_names_dims(dataset)
        self._get_embedding_tables()

        # FM类模型用的first_order_linear
        self.first_order_linear = nn.Linear(self.embedding_size, 1, bias=True)

    def _get_fields_names_dims(self, dataset):
        # 统计各类型特征列名 + token类型特征列有多少种不同值的数量
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.num_feature_field = 0
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue

            if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                self.float_field_names.append(field_name)
            else:
                continue

            self.num_feature_field += 1

    def _get_embedding_tables(self):
        # 给token fields 加偏移，这样便于放到一个统一的embedding_table中
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size)

        # float fields 过一层fc
        if len(self.float_field_names) > 0:
            self.float_embedding_table = nn.Linear(len(self.float_field_names), self.embedding_size)

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """

        # float fields
        float_fields = []
        for field_name in self.float_field_names:
            cur_float_field_values = interaction[field_name].values
            cur_float_field_tensor = torch.from_numpy(cur_float_field_values).unsqueeze(1)
            float_fields.append(cur_float_field_tensor)
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1).float().to(self.device)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # float fields 过一层全连接层转换到self.embedding_size
        dense_embedding = self.embed_float_fields(float_fields)  # [batch_size, embed_dim] or None
        dense_embedding = dense_embedding.unsqueeze(1) if dense_embedding is not None else dense_embedding

        # token fields
        token_fields = []
        for field_name in self.token_field_names:
            cur_token_field_values = interaction[field_name].values
            cur_token_field_tensor = torch.from_numpy(cur_token_field_values).unsqueeze(1)
            token_fields.append(cur_token_field_tensor)
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1).long().to(self.device)  # [batch_size, num_token_field]
        else:
            token_fields = None
        sparse_embedding = self.embed_token_fields(token_fields)  # [batch_size, num_token_field, embed_dim] or None

        # sparse_embedding shape: [batch_size, num_token_field, embed_dim] or None
        # dense_embedding shape:  [batch_size, 1,               embed_dim] or None
        return sparse_embedding, dense_embedding

    def embed_float_fields(self, float_fields: torch.FloatTensor):
        if float_fields is None:
            return None
        float_embedding = self.float_embedding_table(float_fields)
        return float_embedding

    def embed_token_fields(self, token_fields: torch.LongTensor):
        if token_fields is None:
            return None
        token_embedding = self.token_embedding_table(token_fields)
        return token_embedding

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.
        """
        assert self.double_tower

        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)

        if dense_embedding is not None:
            first_dense_embedding = None  # mimic-iii 中，user特征列没有float类型
            second_dense_embedding = dense_embedding
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [self.user_token_field_num, self.item_token_field_num]  # 先用户，后物品，OK
            first_sparse_embedding, second_sparse_embedding = torch.split(sparse_embedding, sizes, dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]


class SequentialEmbeddingLayer(BaseEmbeddingLayer):
    def __init__(self, config, dataset):
        super(SequentialEmbeddingLayer, self).__init__(config, dataset)

        self.user_field_names = dataset.fields(source=[FeatureSource.USER, ])
        self.item_field_names = dataset.fields(source=[FeatureSource.ITEM, ])

        self.USER_ID = config.get("USER_ID_FIELD", "user_id")
        self.ITEM_ID = config.get("ITEM_ID_FIELD", "item_id")
        self.ITEM_SEQ = config.get("HISTORY_ITEM_ID_FIELD", "history")
        self.ITEM_SEQ_LEN = config.get("HISTORY_ITEM_ID_LIST_LENGTH_FIELD", "history_len")
        self.max_seq_length = config.get("MAX_HISTORY_ITEM_ID_LIST_LENGTH", 100)

        self.n_items = dataset.num_items  # 获取有多少个候选物品
        self.item_padding_idx = self.n_items  # 约定padding_idx
        self.item_id_embedding_table = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.item_padding_idx)

        self.user_features = dataset.user_feat_values.to(self.device)
        self.item_features = dataset.item_feat_values.to(self.device)

        self._get_fields_names_dims(dataset)
        self._get_embedding_tables()

    def embed_input_fields(self, user_id, next_item_item_seq):
        user_embedding = self._embed_user_feat_fields(user_id)

        item_seqs_embedding = []
        for ids in next_item_item_seq:
            # L: 当前的item历史序列长度
            L = ids.size(0)
            if L >= self.max_seq_length + 1:  # 过长则截断
                ids = ids[:(self.max_seq_length + 1)]
                valid_ids_feature_embedding = self._embed_item_feat_fields(ids)
            else:  # 过短则填充padding_idx
                # 先获取长度内的有效id的特征emb
                valid_ids_feature_embedding = self._embed_item_feat_fields(ids)

                # 然后填充pad的部分，再和id_embedding相加
                pad_size = (0, 0,
                            0, 0,
                            0, self.max_seq_length + 1 - L)
                valid_ids_feature_embedding = fn.pad(valid_ids_feature_embedding, pad_size, value=0)
                ids = torch.cat([ids,
                                 torch.tensor([self.item_padding_idx] * (self.max_seq_length + 1 - L), device=self.device)])

            ids_embedding = self.item_id_embedding_table(ids).unsqueeze(1)

            item_seqs_embedding.append(torch.cat(
                [valid_ids_feature_embedding, ids_embedding], dim=1).flatten(start_dim=1))

        # num_item_float_field == 0: [B, 1 + max_seq_length, (num_item_token_field + 1) * h]
        # num_item_float_field  > 0: [B, 1 + max_seq_length, (num_item_token_field + 2) * h]
        item_seqs_embedding = torch.stack(item_seqs_embedding)

        return user_embedding, item_seqs_embedding

    def forward(self, interaction):
        user_id      = torch.from_numpy(interaction[self.USER_ID     ].values).to(self.device)
        item_seq_len = torch.from_numpy(interaction[self.ITEM_SEQ_LEN].values).to(self.device)
        item_seq     = interaction[self.ITEM_SEQ].values
        next_items   = interaction[self.ITEM_ID ].values

        collector_next_item_item_seq: List[torch.tensor] = []
        for history_items, next_item in zip(item_seq, next_items):
            # concatenate the history item seq with the target item to get embedding together
            # 注意：这里将target item放在第一个位置，方便后续split
            item_seq_next_item = torch.tensor([next_item] + history_items, device=self.device)
            collector_next_item_item_seq.append(item_seq_next_item)

        user_embedding, item_seqs_embedding = self.embed_input_fields(user_id, collector_next_item_item_seq)
        return user_embedding, item_seqs_embedding


class GraphEmbeddingLayer(nn.Module):
    """异质图中，admission, lab item / drug等结点特征通用的embedding layer；边特征也可以复用"""
    def __init__(self, embedding_size, token_field_dims, float_field_nums):
        super().__init__()
        self.embedding_size = embedding_size
        self.token_field_dims = token_field_dims
        self.float_field_nums = float_field_nums

        # get embedding tables
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims,
                self.token_field_offsets,
                self.embedding_size
            )
        if self.float_field_nums > 0:
            self.float_embedding_table = nn.Linear(self.float_field_nums, self.embedding_size)

    def forward(self, original_features):
        is_batch = len(original_features.size()) > 1
        if not is_batch:
            original_features = original_features.unsqueeze(0)

        # float类型的特征列放最前面，目前药物、检验项目的df都满足要求
        if self.float_field_nums > 0:
            dense_embeddings = self.float_embedding_table(
                original_features[:, :self.float_field_nums].float()
            ).unsqueeze(1)
        else:
            dense_embeddings = None

        if len(self.token_field_dims) > 0:
            sparse_embeddings = self.token_embedding_table(
                original_features[:, -len(self.token_field_dims):].long()
            )
        else:
            sparse_embeddings = None

        if dense_embeddings is not None and sparse_embeddings is not None:
            embeddings = torch.cat([dense_embeddings, sparse_embeddings], dim=1)
        else:
            embeddings = sparse_embeddings if sparse_embeddings is not None else dense_embeddings

        if not is_batch:
            return embeddings.squeeze(0)
        else:
            return embeddings


def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 3个映射矩阵，将最后一维映射到相同维度
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion,
        # shape of queries: (batch_size, no. of queries, 1,         num_hiddens) and
        # shape of keys:    (batch_size, 1, no. of key-value pairs, num_hiddens).
        # Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape.
        # Shape of scores: (batch_size, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 3个映射矩阵，将最后一维映射到相同维度
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion,
        # shape of queries: (batch_size, no. of queries, 1,         num_hiddens) and
        # shape of keys:    (batch_size, 1, no. of key-value pairs, num_hiddens).
        # Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape.
        # Shape of scores: (batch_size, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries:    (batch_size, no. of queries,         d)  # q 和 k 的最后一维需要是一样的！不一样时，使用Additive Attention
    # Shape of keys:       (batch_size, no. of key-value pairs, d)  # k 和 v 的序列长度需要是一样的！
    # Shape of values:     (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)  # residual connect


class AttentionBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, q, k, v, valid_lens):
        Y = self.addnorm1(q, self.attention(q, k, v, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class AttentionEncoder(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(AttentionEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                AttentionBlock(
                    key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, use_bias))

    def forward(self, q, x, valid_lens, *args):
        q = self.pos_encoding(q)
        x = self.pos_encoding(x)

        k, v = x, x
        for i, blk in enumerate(self.blks):
            # 第一层：目标物品为query，历史病情表示为key&value，
            #        用不同的注意力权重，聚合与该物品有关的历史病情表示q
            # 后面层：建模第一层聚合的与该物品有关的历史病情表示q，与历史病情之间的联系
            q = blk(q, k, v, valid_lens)
        return q
