import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, constant_
from typing import List
from model.abstract_recommender import SequentialRecommender
from model.layers import MLPLayers, SequenceAttLayer, SequentialEmbeddingLayer


class DIN(SequentialRecommender):
    def __init__(self, config, dataset):
        super(DIN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size: List = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.device = config["device"]

        # define layers and loss
        self.embedding_layer = SequentialEmbeddingLayer(config, dataset)

        # 第一个+1是因为 conceited id emb
        num_item_feature = len(self.embedding_layer.item_token_field_names) + 1 \
                        + (len(self.embedding_layer.item_float_field_names) > 0)
        num_user_feature = len(self.embedding_layer.user_token_field_names) \
                        + (len(self.embedding_layer.user_float_field_names) > 0)
        self.dnn_list = [3*num_item_feature*self.embedding_size
                         + num_user_feature*self.embedding_size, ] + self.mlp_hidden_size
        self.dnn_mlp_layers = MLPLayers(self.dnn_list, activation="Dice", dropout=self.dropout_prob, bn=True)
        self.att_list = [
            4 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size
        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask
        self.attention = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=False,
            return_seq_weight=False,
        )
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        user_embedding, item_seqs_embedding = self.embedding_layer(interaction)

        target_item_feat_emb, history_item_feat_emd = torch.split(
            item_seqs_embedding, [1, self.max_seq_length], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # attention
        item_seq_len = torch.from_numpy(interaction[self.ITEM_SEQ_LEN].values).to(self.device)
        user_emb = self.attention(target_item_feat_emb, history_item_feat_emd, item_seq_len)
        user_emb = user_emb.squeeze(1)

        # input the DNN to get the prediction score
        din_in = torch.cat(
            [user_embedding, user_emb, target_item_feat_emb, user_emb * target_item_feat_emb], dim=-1
        )
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = torch.from_numpy(interaction[self.LABEL_FIELD].values)\
            .float().to(self.device)
        output = self.forward(interaction)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        scores = self.sigmoid(self.forward(interaction))
        return scores
