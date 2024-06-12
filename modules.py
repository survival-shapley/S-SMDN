
from collections import defaultdict
import itertools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Snake(nn.Module):

    def __init__(self, in_features, a=None, trainable=True):

        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [
            in_features
            ]

        # Initialize `a`
        if a is not None:
            self.a = nn.Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = torch.distributions.Exponential(torch.tensor([0.1]))
            self.a = nn.Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # set the training of `a` to true

    def forward(self, x):

        return  x + (1.0/self.a) * torch.pow(torch.sin(x * self.a), 2)

class MaskedLinear(nn.Linear):
    """
    The method definition is taken from Karpathy
    same as Linear except has a configurable mask on the weights
    """
    def __init__(
            self, in_features, out_features, mask,
            bias=True, dtype=torch.float32
            ):
        super().__init__(in_features, out_features, bias, dtype=dtype)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.set_mask(mask)

    def set_mask(self, mask):
        self.mask.data.copy_(mask.T)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

def create_resulting_mask(input_size, output_size, combinations):
    #Define the last mask
    comb_size = len(combinations)
    comb_range = range(comb_size)
    mask = torch.zeros(input_size, output_size)
    choice = np.random.choice(
        comb_range,
        output_size-comb_size,
        replace=True
        )
    choices = np.sort(np.concatenate([comb_range, choice]))
    for i, choice in enumerate(choices):
        for c in combinations[choice]:
            mask[c,i] = 1
    return mask

def generate_masks(input_size, embedding_size, hidden):
    
    combination_order = 1
    
    features = list(range(input_size))
    combinations = list(
        itertools.combinations(features, combination_order)
        )

    #this is a must in general
    assert all([h>=len(combinations) for h in hidden])

    layers = [input_size] + hidden
    first_mask = create_resulting_mask(
        input_size=layers[0],
        output_size=layers[1],
        combinations=combinations,
        )

    masks = [first_mask]
    layers = list(layers)
    resulting_mask = None
    for h0, h1 in zip(layers[:-1][1:], layers[1:][1:]):
        # print(h0, h1)
        resulting_mask = create_resulting_mask(
            input_size,
            h1,
            combinations
            ) #want to be in the form of

        second_last_mask = []
        for i in range(h1):
            ones = []
            zeros = []
            topk = torch.topk(1*(resulting_mask[:,i] !=0), resulting_mask.size(0))
            for value, index in zip(topk.values, topk.indices):
                if value != 0:
                    ones.append(index.item())
                else:
                    zeros.append(index.item())
            candidate = first_mask[ones].sum(0) != 0
            fltr = first_mask[zeros].sum(0) == 0
            second_last_mask.append((fltr*candidate*1.).view(-1,1))

        second_last_mask = torch.cat(second_last_mask, 1)
        first_mask = torch.matmul(first_mask, second_last_mask)
        masks.append(second_last_mask)
    second_last_mask = []
    if resulting_mask is None:
        resulting_mask = masks[0]
    for row in resulting_mask:
        row = row.view(-1,1).repeat_interleave(embedding_size, -1)
        second_last_mask.append(row)
    second_last_mask = torch.cat(second_last_mask, -1)
    masks.append(second_last_mask)
    return masks

def create_masked_layers(
        d_in, d_hid, d_emb, n_layers, activation, dropout,
        norm=False, dtype=torch.double
        ):

    act_fn = defaultdict(nn.Identity)
    act_fn = {
        'relu':nn.ReLU(),
        'elu':nn.ELU(),
        'selu':nn.SELU(),
        'silu':nn.SiLU(),
        'snake':Snake(d_hid)
        }

    act_fn = act_fn[activation]

    norm_fn = defaultdict(nn.Identity)
    norm_fn.update(
        {
        'layer':nn.LayerNorm(d_hid, eps=1e-5, dtype=dtype),
        'batch':nn.BatchNorm1d(d_hid, eps=1e-5, dtype=dtype),
        }
        )
    norm_fn = norm_fn[norm]

    masks = generate_masks(d_in, d_emb, [d_hid]*(n_layers+1))

    masked_net = list(
        itertools.chain(
            *[
                [
                    MaskedLinear(
                        in_features=mask.size(0),
                        out_features=mask.size(1),
                        mask=mask,
                        dtype=dtype
                    ),
                    act_fn,
                    norm_fn,
                    nn.Dropout(dropout),
                ]
                for ii, mask in enumerate(masks)
            ]
        )
    )
    del masked_net[-3:]

    return masked_net

def create_feedforward_layers(
        d_in, d_hid, d_out, n_layers, activation, dropout, 
        norm, dtype=torch.float32
        ):

    act_fn = defaultdict(nn.Identity)
    act_fn = {
        'relu':nn.ReLU(),
        'elu':nn.ELU(),
        'selu':nn.SELU(),
        'silu':nn.SiLU(),
        'snake':Snake(d_hid)
        }

    act_fn = act_fn[activation]

    norm_fn = defaultdict(nn.Identity)
    norm_fn.update(
        {
        'layer':nn.LayerNorm(d_hid, eps=1e-5, dtype=dtype),
        'batch':nn.BatchNorm1d(d_hid, eps=1e-5, dtype=dtype),
        }
        )
    norm_fn = norm_fn[norm]
    
    neural_net = list(
                itertools.chain(
                    *[
                        [
                            nn.Linear(
                                d_in if ii == 0 else d_hid,
                                d_out if ii + 1 == n_layers else d_hid,
                            ),
                            act_fn,
                            norm_fn,
                            nn.Dropout(dropout * (ii != n_layers - 2)),
                        ]
                        for ii in range(n_layers)
                    ]
                )
            )

    del neural_net[-3:]

    return neural_net
