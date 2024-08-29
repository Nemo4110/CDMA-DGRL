import os
import pandas as pd
import argparse
import torch
import torch.utils.data as torchdata
import utils.constant as constant

from typing import Dict, List
from d2l import torch as d2l
from tqdm import tqdm
from model import context_aware_recommender, general_recommender, sequential_recommender
from dataset.unified import (SourceDataFrames,
                             SingleItemType,
                             SingleItemTypeForContextAwareRec,
                             SingleItemTypeForSequentialRec,
                             DFDataset)
from utils.misc import get_latest_model_ckpt


def get_model_and_dataset_class(model_name):
    if model_name is None:
        raise "INPUT MODEL NAME!"
    elif model_name == "DIN":
        return sequential_recommender.DIN, SingleItemTypeForSequentialRec
    elif model_name == "SASRec":
        return sequential_recommender.SASRec, SingleItemTypeForSequentialRec
    elif model_name == "BPR":
        return general_recommender.BPR, SingleItemType
    elif model_name == "NeuMF":
        return general_recommender.NeuMF, SingleItemType
    elif model_name == "DeepFM":
        return context_aware_recommender.DeepFM, SingleItemTypeForContextAwareRec
    elif model_name == "DSSM":
        return context_aware_recommender.DSSM, SingleItemTypeForContextAwareRec
    else:
        raise NotImplementedError


def prepare_corr_config(model_c, args) -> Dict:
    name_model_c = model_c.__name__
    name_parent_c = model_c.__bases__[0].__name__

    config = {
        "device": torch.device('cuda') if args.use_gpu else torch.device('cpu'),

        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "mlp_hidden_size": [args.hidden_size, args.hidden_size, args.hidden_size],
        "dropout_prob": args.dropout_prob,

        "LABEL_FIELD": "label",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
    }

    if name_parent_c == "GeneralRecommender":
        pass
    elif name_parent_c == "ContextRecommender":
        if name_model_c == "DSSM":
            config["double_tower"] = True
    elif name_parent_c == "SequentialRecommender":
        config["MAX_HISTORY_ITEM_ID_LIST_LENGTH"] = args.max_seq_length
    else:
        raise NotImplementedError

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--goal", default="drug", help="the recommended goal, in ['drug', 'labitem']")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--embedding_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=50)

    parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--epochs", type=int, default=3)  # 不需要——由于训练集非常大，单次遍历足够，且多epochs耗时太长了
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=8192)  # adjustable

    parser.add_argument("--root_path_dataset", default=constant.PATH_MIMIC_III_ETL_OUTPUT)
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub")
    parser.add_argument("--path_dir_results", default=r"./results")
    parser.add_argument("--model_ckpt", default=None)

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sources_dfs = SourceDataFrames(args.root_path_dataset)

    model_class, dataset_class = get_model_and_dataset_class(args.model_name)
    config = prepare_corr_config(model_class, args)

    if args.train:
        train_pre_dataset = dataset_class(sources_dfs, "train", args.goal)
        train_itr_dataset = DFDataset(train_pre_dataset)
        train_dataloader = torchdata.DataLoader(
            train_itr_dataset, batch_size=args.batch_size,
            shuffle=False, pin_memory=True, collate_fn=DFDataset.collect_fn)

        valid_pre_dataset = dataset_class(sources_dfs, "val", args.goal)
        valid_itr_dataset = DFDataset(valid_pre_dataset)
        valid_dataloader = torchdata.DataLoader(
            valid_itr_dataset, batch_size=args.batch_size,
            shuffle=False, pin_memory=True, collate_fn=DFDataset.collect_fn)

        model = model_class(config, train_pre_dataset).to(device)
        path2save = os.path.join(args.path_dir_model_hub, model_class.__bases__[0].__name__, model_class.__name__)
        os.makedirs(path2save, exist_ok=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        min_loss = float("inf")

        train_metric = d2l.Accumulator(2)  # train loss, batch number counter
        model.train()
        train_loop = tqdm(enumerate(train_dataloader), leave=False, ncols=80, total=len(train_dataloader))
        for i, interaction in train_loop:
            loss = model.calculate_loss(interaction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_metric.add(loss.item(), 1)
                train_loop.set_postfix_str(f'train loss: {loss.item():.4f}')

                # 每遍历完训练集的10%或最后一个，在验证集上计算下loss
                if (i > 0 and i % (len(train_dataloader) // 10) == 0) or i == (len(train_dataloader) - 1):
                    valid_metric = d2l.Accumulator(2)
                    model.eval()
                    for val_interaction in valid_dataloader:
                        cur_loss = model.calculate_loss(val_interaction)
                        valid_metric.add(cur_loss.item(), 1)
                        train_loop.set_postfix_str(f'valid loss: {cur_loss.item():.4f}')
                    valid_loss = valid_metric[0] / valid_metric[1]
                    if valid_loss < min_loss:  # 有更小的valid_loss了，保存一下checkpoint
                        min_loss = valid_loss
                        model_name = f"loss_{valid_loss:.4f}_{model.__class__.__name__}_goal_{args.goal}.pt"
                        torch.save(model.state_dict(), os.path.join(path2save, model_name))
                    model.train()  # 退出时恢复下train模式

        print(f"avg. train loss: {train_metric[0] / train_metric[1]:.4f}")

    if args.test:
        test_pre_dataset = dataset_class(sources_dfs, "test", args.goal)
        test_itr_dataset = DFDataset(test_pre_dataset)
        test_dataloader = torchdata.DataLoader(
            test_itr_dataset, batch_size=args.batch_size,
            shuffle=False, pin_memory=True, collate_fn=DFDataset.collect_fn)
        model = model_class(config, test_pre_dataset).to(device)  # 注意这里重新创建了

        if not args.train:
            path2save = os.path.join(args.path_dir_model_hub, model_class.__bases__[0].__name__, model_class.__name__)
            if not args.model_ckpt:
                ckpt_filename = get_latest_model_ckpt(path2save)
            else:
                ckpt_filename = args.model_ckpt
            assert ckpt_filename is not None
            print(f"using saved model: {ckpt_filename}")
            sd_path = os.path.join(path2save, ckpt_filename)
            sd = torch.load(sd_path, map_location=device)
            model.load_state_dict(sd)
        else:
            raise "need model checkpoint!"

        model.eval()
        with torch.no_grad():
            collector: List[pd.DataFrame] = []
            for interaction in tqdm(test_dataloader, leave=False, ncols=80):
                scores = model.predict(interaction)
                interaction['score'] = scores.cpu().tolist()
                collector.append(interaction)

        results: pd.DataFrame = pd.concat(collector, axis=0)
        results.to_csv(os.path.join(args.path_dir_results, f"{ckpt_filename}.csv.gz"), compression='gzip')
