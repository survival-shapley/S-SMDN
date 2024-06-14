
import numpy as np
import pandas as pd

from pycox.datasets import flchain
from pycox.datasets import support
from pycox.datasets import metabric
from pycox.datasets import gbsg

import torch
from torch.utils.data import DataLoader

from auton_lab.auton_survival import datasets, preprocessing 

def one_hot_encode(dataframe, column):
    categorical = pd.get_dummies(dataframe[column], prefix=column)
    dataframe = dataframe.drop(column, axis=1)
    return pd.concat([dataframe, categorical], axis=1, sort=False)

def load_dataset(
        dataset='SUPPORT',
        ):

    if dataset.lower() == 'support_auton':
        outcomes, features = datasets.load_dataset('SUPPORT')
        cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'pbc_auton':
        features, t, e = datasets.load_dataset('PBC')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})
        features = pd.DataFrame(features)
        cat_feats = []
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'support':
        features = support.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])
        features_ = dict()

        feature_names = [
            'age',
            'sex',
            'race',
            'number of comorbidities',
            'presence of diabetes',
            'presence of dementia',
            'presence of cancer',
            'mean arterial blood pressure',
            'heart rate',
            'respiration rate',
            'temperature',
            'white blood cell count',
            "serum’s sodium",
            "serum’s creatinine",
             ]

        for i, key in enumerate(feature_names):
            features_[key] = features.iloc[:,i]
        features = pd.DataFrame.from_dict(features_)

        cat_feats = [
            'sex',
            'race',
            'presence of diabetes',
            'presence of dementia',
            'presence of cancer'
            ]
        num_feats = [
            key for key in features.keys() if key not in cat_feats
            ]

    elif dataset.lower() == 'flchain':
        features = flchain.read_df()
        outcomes = features[['death', 'futime']]
        outcomes = outcomes.rename(
            columns={'death':'event', 'futime':'time'}
            )
        features = features.drop(columns=['death', 'futime'])

        cat_feats = ['flc.grp', 'mgus', 'sex']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'metabric':
        features = metabric.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x4', 'x5', 'x6', 'x7']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'gbsg':
        features = gbsg.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x0', 'x1', 'x2']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'pbc':
        features, t, e = datasets.load_dataset('PBC')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})

        features = pd.DataFrame(features)
        cat_feats = []
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'aki':
        sample_size = 10173
        data = pd.read_csv('./data/ckd.csv').sample(sample_size)
        drop = [x for x in data.keys() if 'unnamed' in x.lower()]
        data = data.drop(columns=drop)
        keep = [
            'age', 
            'Calcium [Mass/volume] in Serum or Plasma.numerical',
            'Chloride [Moles/volume] in Serum or Plasma.numerical',
            'Creatinine [Mass/volume] in Serum or Plasma.numerical',
            'C reactive protein [Mass/volume] in Serum or Plasma.numerical',
            'Potassium [Moles/volume] in Serum or Plasma.numerical',
            'Protein [Mass/volume] in Urine by Test strip.categorical',
            'Urate [Mass/volume] in Serum or Plasma.numerical',
            'time',
            'event'
            ]
        drop = set(data.keys()) - set(keep) 
        # data = data[~data['time'].isna()]

        # let's not use these "pseudo-categorical variables in our experiments"
        pseudo_categorical = [
            key for key in data.keys() if 'categorical' in key
        ]

        data = data.drop(
            columns=[
                        'person_id',
                        'beg_date',
                        'event_date',
                        'cohort_end_date'
                    ] + pseudo_categorical + list(drop)
        )

        data = data.sample(frac=1).reset_index(drop=True)

        outcomes = data[['time', 'event']]
        features = data[[x for x in data.keys() if x not in outcomes.keys()]]

        features = features[~outcomes.time.isna()]
        outcomes = outcomes[~outcomes.time.isna()]
        features = features.loc[:, ~features.columns.duplicated()].copy()
        features = features.fillna(features.mean()).dropna(1)

        num_feats = [key for key in features.keys() if 'numerical' in key]
        num_feats.append('age')
        cat_feats = list(set(features.keys()) - set(num_feats))

    features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
    )

    features = features.astype(np.float32)
    outcomes.time = outcomes.time + 1e-15

    return outcomes, features

class SurvivalData(torch.utils.data.Dataset):
    def __init__(self, x, t, e, bs, device, dtype=torch.double):
        self.bs = bs
        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(t, dtype=dtype),
                torch.tensor(e, dtype=dtype)
            ] for x, t, e in zip(x, t, e)
        ]

        self.device = device
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.device:
                self._cache[index][0] = self._cache[
                    index][0].to(self.device)

                self._cache[index][1] = self._cache[
                    index][1].to(self.device)

                self._cache[index][2] = self._cache[
                    index][2].to(self.device)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_

    def __blen__(self):
        return int(np.ceil(self.__len__() / self.bs))
    
def dataloader(
        features, outcomes, folds, fold,
        tr_size, bs, device
        ):
    
    """
    dtype could be set outside --- however, we will fix it for now.
    """

    x, t, e = features, outcomes.time, outcomes.event
    n = len(features)
    tr_size = int(n * 0.7)

    time_range = np.asarray([t.min(), t.max()])
    horizons = [0.25, 0.5, 0.75]
    quantile_times = np.quantile(t[e == 1], horizons)

    dtype = torch.float
    
    x = features[folds != fold]
    t = outcomes.time[folds != fold]
    e = outcomes.event[folds != fold]

    x_tr, x_val = x[:tr_size], x[tr_size:]
    t_tr, t_val = t[:tr_size], t[tr_size:]
    e_tr, e_val = e[:tr_size], e[tr_size:]

    x_te = features[folds == fold]
    t_te = outcomes.time[folds == fold]
    e_te = outcomes.event[folds == fold]

    loc = t_tr.min(0)
    scale = t_tr.max(0) - t_tr.min(0)

    t_tr = 10 * (t_tr - loc) / scale
    t_val = 10 * (t_val - loc) / scale
    t_te = 10 * (t_te - loc) / scale
    quantile_times = 10 * (quantile_times - loc) / scale
    time_range = 10 * (time_range - loc) / scale

    train_data = SurvivalData(
        x_tr.values, t_tr.values, e_tr.values,
        bs, device, dtype
    )
    valid_data = SurvivalData(
        x_val.values, t_val.values, e_val.values,
        bs, device, dtype
    )
    test_data = SurvivalData(
        x_te.values, t_te.values, e_te.values,
        bs, device, dtype
    )
    
    """
    Outputs dataset of format x, y (i.e., t), c. You might want to 
    standardize w.r.t. time. (
        maybe make in in between 1 and 10 or something
        )
    """
    
    train_dataloader = DataLoader(
        train_data, batch_size=bs, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=bs, shuffle=False
    )
    test_dataloader = DataLoader(
        test_data, batch_size=bs, shuffle=False
    )

    et_tr = np.array(
        [
            (e_tr.values[i], t_tr.values[i]) for i in range(len(e_tr))
        ],
        dtype=[('e', bool), ('t', float)]
    )

    et_val = np.array(
        [
            (e_val.values[i], t_val.values[i]) for i in range(len(e_val))
        ],
        dtype=[('e', bool), ('t', float)]
    )

    et_te = np.array(
        [
            (e_te.values[i], t_te.values[i]) for i in range(len(e_te))
        ],
        dtype=[('e', bool), ('t', float)]
    )

    return (
        train_dataloader, valid_dataloader, test_dataloader, 
        et_tr, et_val, et_te, quantile_times, time_range, horizons
        )
