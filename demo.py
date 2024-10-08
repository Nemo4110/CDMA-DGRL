"""用于跑各个模型的简单测试"""
import torch
import torch.utils.data as torchdata

from d2l import torch as d2l

from model import context_aware_recommender, general_recommender, sequential_recommender
from model.layers import SequentialEmbeddingLayer
from dataset.unified import (SourceDataFrames,
                             SingleItemType,
                             SingleItemTypeForContextAwareRec,
                             SingleItemTypeForSequentialRec,
                             DFDataset)
from utils.enum_type import FeatureType, FeatureSource


if __name__ == '__main__':
    sources_dfs = SourceDataFrames(r"data\mimic-iii-clinical-database-1.4")

    """ context_aware_recommender """
    # dataset = SingleItemTypeForContextAwareRec(sources_dfs, "val", "labitem")

    # DSSM
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "double_tower": True,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "numerical_features": dataset.fields(ftype=[FeatureType.FLOAT])
    # }
    # net = context_aware_recommender.DSSM(config, dataset)

    # DeepFM
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "numerical_features": dataset.fields(ftype=[FeatureType.FLOAT])
    # }
    # net = context_aware_recommender.DeepFM(config, dataset)

    """ GeneralRecommender """
    # dataset = SingleItemType(sources_dfs, "val", "drug")
    # NeuMF
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    # }
    # net = general_recommender.NeuMF(config, dataset)
    # loss = net.calculate_loss(dataset[2])
    # BPR
    # config = {
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    # }
    # net = general_recommender.BPR(config, dataset)
    # loss = net.calculate_loss(dataset[2])

    """ SequentialRecommender """
    # dataset = SingleItemTypeForSequentialRec(sources_dfs, "val", "drug")
    # interaction = dataset[2]

    # SequentialEmbeddingLayer
    # config = {
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    # }
    # emb_layer = SequentialEmbeddingLayer(config, dataset)
    # user_embedding, item_seqs_embedding = emb_layer(interaction)

    # DIN
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    # }
    # net = sequential_recommender.DIN(config, dataset)
    # loss = net.calculate_loss(interaction)
    # print(loss)

    # SASRec
    # config = {
    #     "mlp_hidden_size": [16, 32, 16],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 16,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    # }
    # net = sequential_recommender.SASRec(config, dataset)
    # loss = net.calculate_loss(interaction)
    # print(loss)

    pre_dataset = SingleItemTypeForSequentialRec(sources_dfs, "val", "labitem")
    itr_dataset = DFDataset(pre_dataset)
    itr_dataloader = torchdata.DataLoader(
        itr_dataset, batch_size=4096, shuffle=False, pin_memory=True, collate_fn=DFDataset.collect_fn)

    config = {
        "mlp_hidden_size": [16, 32, 16],
        "dropout_prob": 0.1,
        "embedding_size": 32,
        "device": torch.device('cpu'),
        "LABEL_FIELD": "label",
        "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    }
    net = sequential_recommender.SASRec(config, pre_dataset)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

    for epoch in range(10):
        metric = d2l.Accumulator(2)
        for i, interaction in enumerate(itr_dataloader):
            loss = net.calculate_loss(interaction)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()
            with torch.no_grad():
                metric.add(loss.detach().item(), 1)
                print(f"iter #{i:05}/{len(itr_dataloader):05}, loss: {loss.detach().item():.3f}", flush=True)
        print(f"epoch #{epoch:02}, loss: {metric[0] / metric[1]:02.3f}")
