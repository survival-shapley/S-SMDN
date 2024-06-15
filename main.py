
import os
import argparse

from copy import deepcopy
from collections import defaultdict

import numpy as np
import random

import torch
from torch import optim
from tqdm import tqdm

from datasets import load_dataset, dataloader
from model import Model

from utils import train_one_epoch, evaluate_model, cache_epoch_results,\
    prepare_epoch_results_dict, plot_epoch_results, save_epoch_results,\
    evaluate_survival_horizon, save_fold_results

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     """
#     Dataset and checkpoint arguments.
#     """
#     parser.add_argument(
#         '--dataset', default='support', type=str,
#         help='dataset'
#         )
#     parser.add_argument(
#         '--cv_folds', default=5, type=int, help='cv_folds'
#         )
#     parser.add_argument(
#         '--save_metric', default='LL_valid'
#         )
#     parser.add_argument(
#         '--patience', default=800, type=int, help='cv_folds'
#         )
#     """
#     Device, optimization and batch size arguments.
#     """
#     parser.add_argument(
#         '--device', default='cuda', type=str, help='device to train the model'
#         )
#     parser.add_argument(
#         '--bs', default=1024, type=int, help='batch size'
#         )
#     parser.add_argument(
#         '--lr', default=1e-3, type=float, help='learning rate'
#         )
#     parser.add_argument(
#         '--wd', default=0, type=float, help='weight decay.'
#         )
#     parser.add_argument(
#         '--epochs', default=4000, type=int, help='training epochs number'
#         )
#     """
#     Model arguments.
#     """
#     parser.add_argument(
#         '--d_emb', default=50, type=int, help='missingness embeddings'
#         )
#     parser.add_argument(
#         '--d_clusters', default=5, type=int, help='cluster size'
#         )
#     parser.add_argument(
#         '--d_hid', default=150, type=int, help='hidden dimensions'
#         )
#     parser.add_argument(
#         '--n_layers', default=3, type=int, help='layers in nns'
#         )
#     parser.add_argument(
#         '--act', default='elu', type=str, help='activation function'
#         )
#     parser.add_argument(
#         '--norm', default='layer', type=str, help='layern or batch norm (or none)'
#         )
#     parser.add_argument(
#         '--dropout', default=0.1, type=float, help='dropout'
#         )
#     parser.add_argument(
#         '--beta', default=1, type=float, help='proxy kld scaling coefficient'
#         )
#     args = parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    Dataset and checkpoint arguments.
    """
    parser.add_argument(
        '--dataset', default='aki', type=str,
        help='dataset'
        )
    parser.add_argument(
        '--cv_folds', default=5, type=int, help='cv_folds'
        )
    parser.add_argument(
        '--save_metric', default='LL_valid'
        )
    parser.add_argument(
        '--patience', default=800, type=int, help='cv_folds'
        )
    """
    Device, optimization and batch size arguments.
    """
    parser.add_argument(
        '--device', default='cuda', type=str, help='device to train the model'
        )
    parser.add_argument(
        '--bs', default=1024, type=int, help='batch size'
        )
    parser.add_argument(
        '--lr', default=1e-3, type=float, help='learning rate'
        )
    parser.add_argument(
        '--wd', default=0, type=float, help='weight decay.'
        )
    parser.add_argument(
        '--epochs', default=4000, type=int, help='training epochs number'
        )
    """
    Model arguments.
    """
    parser.add_argument(
        '--d_emb', default=50, type=int, help='missingness embeddings'
        )
    parser.add_argument(
        '--d_clusters', default=3, type=int, help='cluster size'
        )
    parser.add_argument(
        '--d_hid', default=150, type=int, help='hidden dimensions'
        )
    parser.add_argument(
        '--n_layers', default=3, type=int, help='layers in nns'
        )
    parser.add_argument(
        '--act', default='elu', type=str, help='activation function'
        )
    parser.add_argument(
        '--norm', default=None, type=str, help='layern or batch norm (or none)'
        )
    parser.add_argument(
        '--dropout', default=0, type=float, help='dropout'
        )
    parser.add_argument(
        '--beta', default=None, type=float, help='proxy kld scaling coefficient'
        )
    args = parser.parse_args()

    SEED = 12345
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    FLAGS = ', '.join(
        [
            str(y) + ' ' + str(x) for (y,x) in vars(args).items() if y not in [
                'device',
                'dataset',
                'cv_folds'
                ]
            ]
        )

    outcomes, features = load_dataset(args.dataset)

    n = len(features)
    tr_size = int(n * 0.7)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    fold_results = defaultdict(lambda: defaultdict(list))

    """
    If Brier is in save_metric name, then we want to save the model with min
    Brier score.
    """
    criterion  = [min if 'Brier' in args.save_metric else max][0]

    for fold in tqdm(range(args.cv_folds)):

        PATIENCE = 0
        STOP_REASON = 'END OF EPOCHS'

        """
        This is where you prepare the dataset modules.
        Maybe take it to utils.
        """

        train_dataloader, valid_dataloader, test_dataloader, \
        et_tr, et_val, et_te, quantile_times, time_range, horizons\
            = dataloader(
                features, outcomes, folds, fold, tr_size, args.bs, args.device
                )

        d_in = train_dataloader.dataset.input_size()
        model = Model(
            d_in, args.d_clusters, args.d_hid, args.d_emb,
            args.n_layers, args.act, args.norm, args.dropout, args.beta
            ).to(args.device)

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
            )

        epoch_results = prepare_epoch_results_dict(horizons)

        for epoch in range(args.epochs):

            tr_loglikelihood, tr_phi_means = train_one_epoch(
                model,
                optimizer,
                train_dataloader
                )

            val_loglikelihood, val_phi_means, val_survival,\
                val_cis, val_brs, val_roc_auc, val_proxy_kld = evaluate_model(
                    model, valid_dataloader, quantile_times, et_tr, et_val
                    )

            epoch_results = cache_epoch_results(
                epoch_results, tr_loglikelihood, val_loglikelihood,
                val_phi_means, val_cis, val_brs, val_roc_auc, val_proxy_kld,
                fold, epoch, horizons, PATIENCE
                )

            """
            Save best model. If patience, then break. We will plot epoch
            results and move forward to next fold.
            """

            if epoch_results[args.save_metric][-1] == criterion(
                    epoch_results[args.save_metric]
            ):
                print("Caching Best Model...")
                best_model = deepcopy(model)

            if 'Brier' in args.save_metric:
                if epoch_results[args.save_metric][-1] > criterion(
                        epoch_results[args.save_metric]
                        ):
                    PATIENCE += 1
                else:
                    PATIENCE = 0
            else:
                if epoch_results[args.save_metric][-1] < criterion(
                        epoch_results[args.save_metric]
                        ):
                    PATIENCE += 1
                else:
                    PATIENCE = 0
            if PATIENCE >= args.patience:
                print('Early Stopping...')
                STOP_REASON = 'EARLY STOP'
                break
        """
        Here, we evaluate the best model w.r.t. test datalaoder.
        """
        plot_epoch_results(epoch_results, args.dataset, fold, FLAGS)
        save_epoch_results(epoch_results, args.dataset, fold, FLAGS)
        print("\nEvaluating Best Model...")
        test_loglikelihood, test_phi_means, test_survival,\
            test_cis, test_brs, test_roc_auc, test_proxy_kld = evaluate_model(
                best_model, test_dataloader, quantile_times, et_tr, et_te
                )
        ev = evaluate_survival_horizon(
            best_model, test_dataloader, time_range, et_te
            )
        fold_results = save_fold_results(
            fold_results, ev, test_loglikelihood, test_cis, test_brs,
            test_roc_auc, fold, time_range, horizons, args.dataset,
            FLAGS, STOP_REASON
            )
        """
        Save best model.
        """
        os.makedirs('./model_checkpoints', exist_ok=True)
        torch.save(
            best_model,
            './model_checkpoints/{}_fold_{}_{}_({}).pth'.format(
                args.dataset,
                fold,
                'SHAPLEY-MDN',
                FLAGS
                )
            )