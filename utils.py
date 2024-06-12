
from collections import defaultdict


import os

import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv

import torch

import matplotlib.pyplot as plt

from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
    )
    
def to_np(tensor):
    try:
        tensor = torch.detach(tensor).cpu().numpy()
    except:
        tensor = tensor
    return tensor
        
def train_one_epoch(model, optimizer, train_dataloader):
    tr_loglikelihood = []
    tr_phi_means = []
    model.train()
    for x_tr, y_tr, c_tr in train_dataloader:
        optimizer.zero_grad()
        loglikelihood, loss, phi_mean, _ = model(x_tr, y_tr, c_tr)
        tr_phi_means.append(phi_mean)
        (-loss).backward()
        try:
            # if pass it means we are training \beta
            model.beta.grad.data = - model.beta.grad.data
        except:
            # else means \beta is fixed
            pass
        optimizer.step()
        tr_loglikelihood.append(loglikelihood.item())
    tr_phi_means = to_np(torch.mean(torch.stack(tr_phi_means), 0))
    tr_loglikelihood = np.mean(tr_loglikelihood)
    return tr_loglikelihood, tr_phi_means

def evaluate_model(
        model, dataloader, times, train, valid,
        ):
    with torch.no_grad():
        val_loglikelihood = []
        val_phi_means = []
        val_survival = []
        val_proxy_kld = []
        model.eval()
        for x, y, c in dataloader:
            loglikelihood, loss, phi_mean, proxy_kld = model(x, y, c)
            survival = model.survival_times(x, times)
            val_phi_means.append(phi_mean)
            val_survival.append(survival)
            val_loglikelihood.append(loglikelihood.item())
            val_proxy_kld.append(proxy_kld.mean().item())
        val_phi_means = to_np(torch.mean(torch.stack(val_phi_means), 0))
        val_loglikelihood = np.mean(val_loglikelihood)
        val_proxy_kld = np.mean(val_proxy_kld)
        val_survival = pd.concat(val_survival)
        survival = val_survival.values
        risk = 1 - survival

        val_cis = []
        val_brs = []
        for i, _ in enumerate(times):
            val_cis.append(
                concordance_index_ipcw(
                    train, valid, risk[:, i], times[i]
                )[0]
            )

        max_val = max([k[1] for k in valid])
        max_tr = max([k[1] for k in train])
        while max_val > max_tr:
            idx = [k[1] for k in valid].index(max_val)
            valid = np.delete(valid, idx, 0)
            survival = np.delete(survival, idx, 0)
            risk = np.delete(risk, idx, 0)
            max_val = max([k[1] for k in valid])

        val_brs.append(
            brier_score(
                train, valid, survival, times
            )[1]
        )

        val_roc_auc = []
        for i, _ in enumerate(times):
            val_roc_auc.append(
                cumulative_dynamic_auc(
                    train, valid, risk[:, i], times[i]
                )[0]
            )

    return val_loglikelihood, val_phi_means, val_survival,\
        val_cis, val_brs, val_roc_auc, val_proxy_kld

def evaluate_survival_horizon(best_model, test_dataloader, time_range, et_te):
    with torch.no_grad():
        surv_df = []
        times = []
        censors = []
        for (x,y,c) in test_dataloader:
            surv = best_model.survival_horizon(
                x, time_range[0], time_range[1], 100
                )
            surv_df.append(surv)
            times.append(y)
            censors.append(c)
        surv_df = pd.concat(surv_df)
        surv_df = surv_df.reset_index(drop=True)
        ev = EvalSurv(
            surv_df,
            np.asarray([t[1] for t in et_te]),
            np.asarray([t[0] for t in et_te]),
            censor_surv='km'
            )
    return ev
    

def cache_epoch_results(
                epoch_results,
                tr_loglikelihood, val_loglikelihood, val_phi_means,
                val_cis, val_brs, val_roc_auc, val_proxy_kld,
                fold, epoch, horizons,
                PATIENCE
                ):
    print(
        "\nFold: {} Epoch: {}, LL_train: {}, LL_valid: {}".format(
            fold,
            epoch,
            round(tr_loglikelihood, 6),
            round(val_loglikelihood, 6),
            )
        )
    
    epoch_results['LL_train'].append(tr_loglikelihood)
    epoch_results['LL_valid'].append(val_loglikelihood)
    epoch_results['PHI_means'].append(val_phi_means)
    # epoch_results['KLD'].append(val_proxy_kld)
    
    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index:", val_cis[horizon[0]])
        print("Brier Score:", val_brs[0][horizon[0]])
        print("ROC AUC:", val_roc_auc[horizon[0]][0], "\n")
        epoch_results[
            'C-Index {} quantile'.format(horizon[1])
        ].append(val_cis[horizon[0]])
        epoch_results[
            'Brier Score {} quantile'.format(horizon[1])
        ].append(val_brs[0][horizon[0]])
        epoch_results[
            'ROC AUC {} quantile'.format(horizon[1])
        ].append(val_roc_auc[horizon[0]][0])

    print('Patience: {}'.format(PATIENCE))
    
    return epoch_results

def prepare_epoch_results_dict(horizons):
    
    epoch_results = defaultdict(list)
    
    epoch_results['LL_train'].append(-np.inf)
    epoch_results['LL_valid'].append(-np.inf)
    epoch_results['PHI_means'].append(np.inf)

    for horizon in enumerate(horizons):

        epoch_results[
            'C-Index {} quantile'.format(horizon[1])
        ].append(-np.inf)
        epoch_results[
            'Brier Score {} quantile'.format(horizon[1])
        ].append(np.inf)
        epoch_results[
            'ROC AUC {} quantile'.format(horizon[1])
        ].append(-np.inf)
    
    return epoch_results

def plot_epoch_results(epoch_results, dataset, fold, FLAGS):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 5))
    ax[0][0].plot(epoch_results['LL_train'], color='b', label="LL_train")
    ax[0][0].plot(epoch_results['LL_valid'], color='r', label="LL_valid")
    ax[0][0].legend()
    ax[0][0].set_xlabel('Epochs')
    color = ['r', 'g', 'b']
    i = 0
    j = 0
    k = 0
    for (key, value) in epoch_results.items():
        if 'C-Index' in key:
            ax[0][1].plot(value, color=color[i], label=key)
            ax[0][1].legend()
            i += 1
        elif 'Brier' in key:
            ax[1][0].plot(value, color=color[j], label=key)
            ax[1][0].legend()
            j += 1
        elif 'ROC' in key:
            ax[1][1].plot(value, color=color[k], label=key)
            ax[1][1].legend()
            k += 1
    ax[0][1].set_xlabel('Epochs')
    ax[1][0].set_xlabel('Epochs')
    ax[1][1].set_xlabel('Epochs')
    plt.tight_layout()
    os.makedirs('./fold_figures', exist_ok=True)
    plt.savefig("./fold_figures/{}_fold_{}_{}_figs_({}).svg".format(
            dataset,
            fold,
            'SHAPLEY-MDN',
            FLAGS
            )
        )

def save_epoch_results(epoch_results, dataset, fold, FLAGS):
    epoch_results = pd.DataFrame(epoch_results)
    os.makedirs('./epoch_results', exist_ok=True)
    epoch_results.to_csv(
        './epoch_results/{}_fold_{}_{}_epoch_res_({}).csv'.format(
            dataset,
            fold,
            'SHAPLEY-MDN',
            FLAGS
            )
        )
                
def save_fold_results(
        fold_results, ev, test_loglikelihood,
        test_cis, test_brs, test_roc_auc,
        fold, time_range, horizons, dataset, FLAGS, STOP_REASON
        ):
    print("\nTest Loglikelihood: {}".format(test_loglikelihood))
    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index:", test_cis[horizon[0]])
        print("Brier Score:",test_brs[0][horizon[0]])
        print("ROC AUC ", test_roc_auc[horizon[0]][0], "\n")

        fold_results[
            'Fold: {}'.format(fold)
        ][
            'C-Index {} quantile'.format(horizon[1])
        ].append(test_cis[horizon[0]])
        fold_results[
            'Fold: {}'.format(fold)
        ][
            'Brier Score {} quantile'.format(horizon[1])
        ].append(test_brs[0][horizon[0]])
        fold_results[
            'Fold: {}'.format(fold)
        ][
            'ROC AUC {} quantile'.format(horizon[1])
        ].append(test_roc_auc[horizon[0]][0])
    """
    fold_results[
        'Fold: {}'.format(fold)
    ][
        'Integrated Brier Score'
    ].append(
        ev.brier_score(
            np.linspace(time_range[0], time_range[1], 100)
            ).mean()
        )

    fold_results[
        'Fold: {}'.format(fold)
    ][
        'Antolini C-Index'
    ].append(ev.concordance_td('antolini'))

    fold_results[
        'Fold: {}'.format(fold)
    ][
        'Integrated NBLL'
    ].append(
        ev.integrated_nbll(
            np.linspace(time_range[0], time_range[1], 100)
            ).mean()
        )
    """
    fold_results[
        'Fold: {}'.format(fold)
    ][
        'Stop Reason'
    ].append(STOP_REASON)
    fold_results_ = pd.DataFrame(fold_results)
    for key in fold_results_.keys():
        fold_results_[key] = [
            _[0] for _ in fold_results_[key]
        ]
    os.makedirs('./fold_results', exist_ok=True)
    fold_results_.to_csv(
        './fold_results/{}_{}_fold_results_({}).csv'.format(
            dataset,
            'SHAPLEY-MDN',
            FLAGS
            )
        )
    return fold_results
