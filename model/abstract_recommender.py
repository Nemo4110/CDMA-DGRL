import torch
import torch.nn as nn
import numpy as np

from dataset.unified import SingleItemType
from utils.misc import set_color
from model.layers import GeneralEmbeddingLayer, ContextEmbeddingLayer


class AbstractRecommender(nn.Module):
    def __init__(self):
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data."""
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items."""
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )


class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # 获取数据集信息
        self.USER_ID = config.get("USER_ID_FIELD", "user_id")  # 记录用户id的列名
        self.ITEM_ID = config.get("ITEM_ID_FIELD", "item_id")  # 记录物品id的列名

        self.embedding_size = config["embedding_size"]
        self.embedding_layer = GeneralEmbeddingLayer(config, dataset)

        self.device = config["device"]

        self.num_user_feature = len(self.embedding_layer.user_token_field_names) + 1 \
            if len(self.embedding_layer.user_float_field_names) > 0 \
            else len(self.embedding_layer.user_token_field_names)
        self.num_item_feature = len(self.embedding_layer.item_token_field_names) + 1 \
            if len(self.embedding_layer.item_float_field_names) > 0 \
            else len(self.embedding_layer.item_token_field_names)

        # 用于对齐维度
        self.uf_aligner = nn.Linear(self.num_user_feature * self.embedding_size, self.embedding_size)
        self.if_aligner = nn.Linear(self.num_item_feature * self.embedding_size, self.embedding_size)


class SequentialRecommender(AbstractRecommender):
    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]

        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]


class ContextRecommender(AbstractRecommender):
    def __init__(self, config, dataset: SingleItemType):
        super(ContextRecommender, self).__init__()
        self.device = config["device"]
        self.embedding_size = config["embedding_size"]
        self.LABEL = config["LABEL_FIELD"]

        self.embedding_layer = ContextEmbeddingLayer(config, dataset)
