import argparse
import os
import pandas as pd
import torch
import utils.constant as constant

from d2l import torch as d2l
from typing import List
from tqdm import tqdm

from dataset.unified import SourceDataFrames, OneAdmOneHG
from model.backbone import BackBoneV2
from utils.misc import get_latest_model_ckpt, EarlyStopper
from utils.config import HeteroGraphConfig, GNNConfig
from utils.metrics import convert2df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # following arguments are model settings
    parser.add_argument("--gnn_type",                                 default="GENConv")
    parser.add_argument("--gnn_layer_num",                type=int,   default=3)
    parser.add_argument("--num_encoder_layers",           type=int,   default=3)
    parser.add_argument("--hidden_dim",                   type=int,   default=256)
    parser.add_argument("--embedding_size",               type=int,   default=10)
    parser.add_argument("--lr",                           type=float, default=0.001)

    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_ETL_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")

    # Experiment settings
    parser.add_argument("--item_type",                              default="MIX")
    parser.add_argument("--goal",                                   default="drug",     help="the goal of the recommended task, in ['drug', 'labitem']")
    parser.add_argument("--is_gnn_only",      action="store_true",  default=False,      help="whether to only use GNN")
    parser.add_argument("--train",            action="store_true",  default=False)
    parser.add_argument("--patience",         type=int,             default=3)
    parser.add_argument("--test",             action="store_true",  default=False)
    parser.add_argument("--test_model_state_dict",                  default=None,      help="test only model's state_dict file name")  # must be specified when --train=False!
    parser.add_argument("--model_ckpt",                             default=None,      help="the .pt filename where stores the state_dict of model")
    parser.add_argument("--use_gpu",          action="store_true",  default=False)

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sources_dfs = SourceDataFrames(args.root_path_dataset)

    if args.item_type == "MIX":
        node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    else:
        node_types, edge_types = HeteroGraphConfig.use_one_edge_type(item_type=args.item_type)
    gnn_conf = GNNConfig(args.gnn_type, args.gnn_layer_num, node_types, edge_types)
    model = BackBoneV2(sources_dfs, args.goal, args.hidden_dim, gnn_conf, device,
                       args.num_encoder_layers, args.embedding_size, args.is_gnn_only).to(device)
    os.makedirs(args.path_dir_model_hub, exist_ok=True)

    if args.train:
        train_dataset = OneAdmOneHG(sources_dfs, "train")
        valid_dataset = OneAdmOneHG(sources_dfs, "val")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        min_loss = float("inf")
        early_stopper = EarlyStopper(args.patience, False)
        if args.use_gpu:
            torch.cuda.empty_cache()
        train_metric = d2l.Accumulator(2)  # train loss, iter num
        model.train()
        train_loop = tqdm(enumerate(train_dataset), leave=False, ncols=80, total=len(train_dataset))
        for i, hg in train_loop:
            hg = hg.to(device)
            logits, labels = model(hg)
            loss = BackBoneV2.get_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                train_metric.add(loss.detach().item(), 1)
                train_loop.set_postfix_str(f'train loss: {loss.item():.4f}')

                # 每遍历完训练集的10%或最后一个，在验证集上计算下loss
                if (i > 0 and i % (len(train_dataset) // 10) == 0) or i == (len(train_dataset) - 1):
                    model.eval()
                    valid_metric = d2l.Accumulator(2)
                    for hg in valid_dataset:
                        hg = hg.to(device)
                        logits, labels = model(hg)
                        validloss = BackBoneV2.get_loss(logits, labels)
                        valid_metric.add(validloss.item(), 1)
                        train_loop.set_postfix_str(f'valid loss: {validloss.item():.4f}')

                    valid_loss = valid_metric[0] / valid_metric[1]
                    early_stopper(valid_loss)
                    # 在没有早停的前提下
                    if not early_stopper.early_stop:
                        # 有更小的valid_loss，或到最后了且valid_loss更小了
                        if valid_loss < min_loss or (i == (len(train_dataset) - 1) and valid_loss < min_loss):
                            min_loss = min(min_loss, valid_loss)
                            model_name = f"loss_{valid_loss:.4f}_{model.__class__.__name__}_goal_{args.goal}.pt"
                            torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub, model_name))
                    else:
                        print("Early stop due to increasing valid loss!")
                        break  # 早停
                    model.train()

        print(f"avg. train loss: {train_metric[0] / train_metric[1]:.4f}")

    if args.test:
        test_dataset = OneAdmOneHG(sources_dfs, "test")

        if not args.train:
            # auto load latest save model from hub
            if not args.model_ckpt:
                ckpt_filename = get_latest_model_ckpt(args.path_dir_model_hub)
            else:
                ckpt_filename = args.model_ckpt
            assert ckpt_filename is not None
            print(f"using saved model: {ckpt_filename}")
            sd_path = os.path.join(args.path_dir_model_hub, ckpt_filename)
            sd = torch.load(sd_path, map_location=device)
            model.load_state_dict(sd)
        else:
            ckpt_filename = model_name

        model.eval()
        with torch.no_grad():
            collector: List[pd.DataFrame] = []
            for hg in tqdm(test_dataset, leave=False, ncols=80, total=len(test_dataset)):
                hg = hg.to(device)
                logits, labels = model(hg)

                # 把预测结果全部收集成DataFrame，后面再单独写notebook/脚本进行细致的指标计算
                collector.append(convert2df(logits, labels))

        results: pd.DataFrame = pd.concat(collector, axis=0)
        results.to_csv(os.path.join(args.path_dir_results, f"{ckpt_filename}.csv.gz"), compression='gzip')
