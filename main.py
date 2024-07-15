import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.constant as constant

from tqdm import tqdm
from dataset.hgs import DiscreteTimeHeteroGraph
from model.backbone import BackBone
from model.seq_recommend import SeqRecommend
from utils.metrics import Logger
from utils.best_thresholds import BestThreshldLogger
from utils.misc import calc_loss, node_type_to_prefix, get_latest_model_ckpt
from utils.config import HeteroGraphConfig, GNNConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings

    # NOTE: when max_timestep set to 30 or 50,
    #       would trigger the assert error "last timestep has not labels!"
    #       in `get_subgraph_by_timestep` (bigger max_timestep can be support in future)
    parser.add_argument("--max_timestep",                 type=int,   default=20,                   help="The maximum `TIMESTEP`")
    parser.add_argument("--gnn_type",                                 default="GENConv")
    parser.add_argument("--gnn_layer_num",                type=int,   default=2)
    parser.add_argument("--num_decoder_layers",           type=int,   default=6)
    parser.add_argument("--decoder_choice",                           default="TransformerDecoder")
    parser.add_argument("--hidden_dim",                   type=int,   default=64)
    parser.add_argument("--lr",                           type=float, default=0.0003)
    parser.add_argument("--use_seq_rec",      action="store_true",    default=False,                help="whether to use sequntial recommendation (without GNN)")
    parser.add_argument("--is_gnn_only",      action="store_true",    default=False,                help="whether to only use GNN")
    parser.add_argument("--is_seq_pred",      action="store_true",    default=False,                help="whether to enable seq pred")

    # Paths
    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_HGS_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")
    parser.add_argument("--path_dir_thresholds",default=r"./thresholds",                    help="path where thresholds save")

    # Experiment settings
    parser.add_argument("--task",                                   default="MIX",     help="the goal of the recommended task, in ['MIX', 'drug', 'labitem']")
    parser.add_argument("--epochs",                       type=int, default=10)
    parser.add_argument("--train",            action="store_true",  default=False)

    parser.add_argument("--test",             action="store_true",  default=False)
    parser.add_argument("--test_model_state_dict",                  default=None,      help="test only model's state_dict file name")  # must be specified when --train=False!
    parser.add_argument("--model_ckpt",                             default=None,      help="the .pt filename where stores the state_dict of model")
    parser.add_argument("--test_num",                     type=int, default=-1,        help="number of testing")

    parser.add_argument("--use_gpu",          action="store_true",  default=False)
    parser.add_argument("--batch_size",                   type=int, default=16)
    parser.add_argument("--batch_size_by_HADMID",         type=int, default=128,       help="specified the batch size that will be used for splitting the dataset by HADM_ID")
    parser.add_argument("--neg_smp_strategy",             type=int, default=0,         help="the stratege of negative sampling")

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # 数据集位置
    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")

    # Heterogeneous graph config
    if args.task == "MIX":
        node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    else:
        node_types, edge_types = HeteroGraphConfig.use_one_edge_type(item_type=args.task)

    # 设置model
    if not args.use_seq_rec:
        model: nn.Module = BackBone(
            max_timestep=args.max_timestep,
            gnn_type=args.gnn_type,
            gnn_layer_num=args.gnn_layer_num,
            is_gnn_only=args.is_gnn_only,
            node_types=node_types,
            edge_types=edge_types,
            decoder_choice=args.decoder_choice,
            num_decoder_layers=args.num_decoder_layers,
            hidden_dim=args.hidden_dim,
            neg_smp_strategy=args.neg_smp_strategy
        ).to(device)
    else:
        model: nn.Module = SeqRecommend(
            max_timestep=args.max_timestep,
            neg_smp_strategy=args.neg_smp_strategy,
            node_types=node_types,
            edge_types=edge_types,
            decoder_choice=args.decoder_choice,
            num_layers=args.num_decoder_layers,
            hidden_dim=args.hidden_dim,
            is_seq_pred=args.is_seq_pred
        ).to(device)

    # --- train ---
    if args.train:
        model.train()
        if not os.path.exists(args.path_dir_thresholds): os.mkdir(args.path_dir_thresholds)
        best_threshold_loggers = {
            node_type: BestThreshldLogger(max_timestep=args.max_timestep, save_dir_path=args.path_dir_thresholds)
            for node_type in node_types if node_type != 'admission'
        }
        loss_f = F.binary_cross_entropy_with_logits

        train_set = DiscreteTimeHeteroGraph(root_path=root_path, usage="train")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # train
        # with torch.autograd.detect_anomaly():
        for epoch in tqdm(range(args.epochs)):
            if args.use_gpu:
                torch.cuda.empty_cache()

            t_loop_train_set = tqdm(train_set, leave=False)
            for hg in t_loop_train_set:
                hg = hg.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    dict_every_day_pred = model(hg)
                    loss = calc_loss(dict_every_day_pred, node_types, device, loss_f)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()  # optimizing

                # log the best_threshold
                if epoch == (args.epochs - 1):
                    for node_type, best_threshold_logger in best_threshold_loggers.items():
                        best_threshold_logger.log(dict_every_day_pred[node_type]["scores"],
                                                  dict_every_day_pred[node_type]["labels"])

                t_loop_train_set.set_postfix_str(f'\033[32m Current loss: {loss.item():.4f} \033[0m')

        # save the best_threshold
        for node_type, best_threshold_logger in best_threshold_loggers.items():
            best_threshold_logger.save(
                prefix=f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")

        # save trained model
        model_saving_prefix = f"task={args.task}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
        if not os.path.exists(args.path_dir_model_hub):
            os.mkdir(args.path_dir_model_hub)
        torch.save(model.state_dict(),
                   os.path.join(args.path_dir_model_hub, f"{model_saving_prefix}_loss={loss:.4f}.pt"))

    # --- test ---
    if args.test:
        resl_path = os.path.join(args.path_dir_results, f"#{args.test_num}")
        if not os.path.exists(resl_path):
            os.mkdir(resl_path)

        test_set = DiscreteTimeHeteroGraph(root_path=root_path, usage="test")

        if not args.train:
            # TODO: auto load last checkpoint
            model_state_dict = torch.load(os.path.join(args.path_dir_model_hub, f"{args.test_model_state_dict}.pt"),
                                          map_location=device)
            model.load_state_dict(model_state_dict)

        # metrics loggers
        metrics_loggers = {}
        for node_type in node_types:
            if node_type == 'admission':
                continue
            metrics_loggers[node_type] = Logger(max_timestep=args.max_timestep,
                                                save_dir_path=resl_path,
                                                best_thresholdspath=os.path.join(args.path_dir_thresholds,
                                                                                 f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}_best_thresholds.pickle"),
                                                is_calc_ddi=True if node_type == 'drug' else False)

        model.eval()
        with torch.no_grad():
            for hg in tqdm(test_set):
                hg = hg.to(device)
                dict_every_day_pred = model(hg)

                for node_type, metrics_logger in metrics_loggers.items():
                    metrics_logger.log(dict_every_day_pred[node_type]["scores"],
                                       dict_every_day_pred[node_type]["labels"],
                                       dict_every_day_pred[node_type]["indices"] if node_type == 'drug' else None)

        for node_type, metrics_logger in metrics_loggers.items():
            metrics_logger.save(
                description=f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")
