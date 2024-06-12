
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.distributions.normal import Normal

from modules import create_feedforward_layers, create_masked_layers

from utils import to_np

class SimplexNet(nn.Module):

    def __init__(
            self, d_in, d_out, d_hid, n_layers, activation, dropout, norm
            ):
        super(SimplexNet, self).__init__()

        self.pi_network = nn.Sequential(
            nn.Sequential(
                *create_feedforward_layers(
                    d_in=d_in, d_hid=d_hid, d_out=d_out,
                    n_layers=n_layers, activation=activation,
                    dropout=dropout, norm=norm
                    )
                )
            )

    def forward(self, phi):

        logprobs = nn.LogSoftmax(dim=-1)(self.pi_network(phi))
        probs = logprobs.exp()

        return probs, logprobs

class PredictiveNet(nn.Module):

    def __init__(
            self, d_in, d_out, d_hid, d_emb, d_clusters, n_layers,
            activation, dropout, norm
            ):
        super(PredictiveNet, self).__init__()

        self.loc_net = nn.Sequential(
            *create_feedforward_layers(
                d_in + d_emb, d_hid, d_out, n_layers, activation, dropout, norm
                )
            )

        self.scale_net = nn.Sequential(
            nn.Sequential(
                *create_feedforward_layers(
                d_in + d_emb, d_hid, d_out, n_layers, activation, dropout, norm
                )
                )
            )

        self.cluster_emb = nn.Embedding(d_clusters, d_emb)
        self.d_clusters = d_clusters

    def forward(self, phi):

        c = torch.tensor(list(range(self.d_clusters))).long().to(phi.device)
        c = c.repeat_interleave(phi.size(0), 0)

        phi = torch.cat([phi for _ in range(self.d_clusters)], 0)
        c_emb = self.cluster_emb(c)

        loc = self.loc_net(
            torch.cat([phi, c_emb], -1)
            ).squeeze(-1)
        scale = self.scale_net(
            torch.cat([phi, c_emb], -1)
            ).squeeze(-1)
        scale = nn.Softplus()(scale)

        return loc, scale + 1e-10

class ShapleyNet(nn.Module):
    """
    Given a feature, outputs Shapley values.
    """
    def __init__(
            self, d_in, d_clusters, d_hid,
            d_emb, n_layers, activation, norm, dropout
            ):
        super(ShapleyNet, self).__init__()

        self.placeholder = nn.Parameter(
            torch.zeros(d_in, d_emb), requires_grad=True
            )

        self.cont_emb_loc = nn.Sequential(
            *create_masked_layers(
                d_in=d_in, d_hid=d_hid, d_emb=d_emb, n_layers=n_layers-2,
                activation=activation, dropout=dropout,
                norm=None, dtype=torch.float32
                )
            )

        self.cont_emb_shapley = nn.Sequential(
            *create_masked_layers(
                d_in=d_in, d_hid=d_hid, d_emb=d_emb, n_layers=n_layers-2,
                activation=activation, dropout=dropout,
                norm=None, dtype=torch.float32
                )
            )

        self.loc_net = nn.Sequential(
            *create_feedforward_layers(
                d_emb, d_hid, d_in, n_layers, activation, dropout, norm
                )
            )

        self.combine_loc = nn.Linear(d_in, 1, bias=False)

    def forward(self, x, m):

        x_emb_loc = self.cont_emb_loc(x).view(x.size(0), x.size(-1), -1)
        m_ = m.repeat_interleave(
            x_emb_loc.size(-1),1
            ).view(x.size(0), x.size(-1),-1)
        x_emb_loc = (
            x_emb_loc.masked_fill(m_ == 0, 0) + (
                self.placeholder * (1 - m_ * 1.)
                )
            )
        x_emb_loc = x_emb_loc.view(x.size(0),-1)
        x_emb_loc = self.combine_loc(
            x_emb_loc.view(
                x_emb_loc.size(0), m.size(1), -1
                ).transpose(-1,-2)
            ).squeeze(-1)

        loc = self.loc_net(x_emb_loc)

        return loc.squeeze(-1)

##############################################################################
##############################################################################

class Model(nn.Module):
    def __init__(
            self, d_in, d_clusters, d_hid, d_emb,
            n_layers, activation, norm, dropout, beta
            ):
        super(Model, self).__init__()

        self.simplex_net = SimplexNet(
            d_in, d_clusters, d_hid, n_layers, activation, dropout, norm
            )
        self.predictive_net = PredictiveNet(
             d_in, 1, d_hid, d_emb, d_clusters, n_layers,
             activation, dropout, norm
            )
        self.shapley_net = ShapleyNet(
                d_in, d_clusters, d_hid,
                d_emb, n_layers, activation, norm, dropout
                )
        self.d_clusters = d_clusters

        if beta is None:
            self.beta = nn.Parameter(
                torch.randn(1),
                requires_grad=True
                )
        else:
            self.beta = nn.Parameter(
                torch.tensor([np.log(beta)]),
                requires_grad=False
                )

    def compute_loc_scale(self, x, m):

        phi = self.shapley_net(x, m)
        pc_x, logpc_x = self.simplex_net(phi)
        predictive_loc, predictive_scale = self.predictive_net(phi)
        predictions = torch.sum(
            torch.cat(
                [pl.unsqueeze(-1) for pl in predictive_loc.split(x.size(0),0)],
                -1
                ) * pc_x,
            -1
            )

        return phi, predictive_loc, predictive_scale, predictions

    def joint_and_posterior(
            self, logpc_x, predictive_loc, predictive_scale, y, c
            ):

        predictive_loc = torch.cat(
            [pl.unsqueeze(-1) for pl in predictive_loc.split(y.size(0),0)],
            -1
            )
        predictive_scale = torch.cat(
            [ps.unsqueeze(-1) for ps in predictive_scale.split(y.size(0),0)],
            -1
            )

        y = y.unsqueeze(-1).repeat_interleave(self.d_clusters, -1)
        c = c.unsqueeze(-1).repeat_interleave(self.d_clusters, -1)

        logjoint = c * (
            logpc_x + Normal(
                loc=predictive_loc, scale=predictive_scale
                ).log_prob(y)
            )

        logjoint += (1-c) * (
            logpc_x + (1 - Normal(
                loc=predictive_loc, scale=predictive_scale
                ).cdf(y) + 1e-10).log()
            )

        # Compute logposterior and posterior.
        # Since we use EM here, we detach them. This is optional, change the
        # line below to q_function if you use EM algorithm

        logposterior = torch.detach(
            logjoint - logjoint.logsumexp(-1).view(-1,1)
            )

        posterior = logposterior.exp()
        return logjoint, posterior, logposterior

    def forward(self, x, y, c):

        m = self.missingness_indicator(x)
        phi, predictive_loc, predictive_scale, \
            predictions = self.compute_loc_scale(x, m)
        pc_x, logpc_x = self.simplex_net(phi)

        phi_mean = (predictions - predictions.mean() - phi.sum(-1))
        phi_mean = phi_mean.mean(0).pow(2)

        logjoint, posterior, \
            logposterior = self.joint_and_posterior(
                logpc_x, predictive_loc, predictive_scale, y, c
                )
        # in case, you want to use EM-Algorithm
        # q_function = torch.sum(posterior * logjoint, -1)
        # if you want to use EM replace loglikelihood in loss with q_function
        # by de-commenting q_function
        loglikelihood = logjoint.logsumexp(-1)

        posterior_loc1, posterior_loc2, feature_idx \
            = self.posterior_parameters(x, m)

        phi = torch.cat(
            [p.gather(
            1,feature_idx.long().unsqueeze(-1)
            )
        for p in phi.split(x.size(0), 0)],-1).squeeze(-1)

        # beta * (delta_blue.pow(2) + delta_red.pow(2)).detach() / 2 is only
        # important if you are defining a min max game
        # (and hence, training \beta). proxy kld is a regularizer
        # that establishes and unbiased estimator to the gradients of actual kld
        # it can sometimes be negative (unlike true kld). hence, to have a better
        # estimate for the gradients w.r.t. \beta, we add this pseudo-term
        # this (min-max) goes to the appendix not the main paper

        beta = self.beta.exp()
        delta_red = posterior_loc2 - phi
        delta_blue = posterior_loc1 - phi
        proxy_kld = beta.detach() * (
            delta_blue * delta_red.detach()
            ) + beta * (delta_blue.pow(2) + delta_red.pow(2)).detach() / 2

        loss = torch.mean(
            loglikelihood - proxy_kld
            , 0)
        loglikelihood = torch.mean(loglikelihood, 0)
        proxy_kld = torch.mean(proxy_kld, 0)
        return loglikelihood, loss, phi_mean.pow(0.5), proxy_kld

    def missingness_indicator(self, x):

        if self.training:
            feature_idx_init = torch.tensor(
                np.random.choice(range(x.size(-1)), x.size(0))
                ).to(x.device)
            feature_idx = feature_idx_init.unsqueeze(-1)

            permutation = torch.argsort(
                torch.rand(
                    x.size(0),
                    x.size(-1)),
                dim=-1
                ).to(x.device)

            arange = torch.arange(x.size(-1)).unsqueeze(0).repeat_interleave(
                permutation.size(0), 0
                ).to(x.device)
            pointer = arange <= torch.argmax(
                (permutation == feature_idx) * 1., -1
                ).view(-1,1)
            p_sorted = (-permutation).topk(
                permutation.size(-1), -1, sorted=True
                )[1]
            m = torch.cat(
                [
                    torch.diag(
                        pointer[:,p_sorted[:,i]]
                        ).view(-1,1) for i in range(
                            p_sorted.size(-1)
                            )
                ], -1
                            )

        else:
            m = torch.ones_like(x) == 1
        return m

    def posterior_parameters(
            self,
            x,
            missing_,
            sample_size=1
            ):
        sample_size *= 2
        orig_size = x.size(0)
        feature_idx_init = torch.tensor(
            np.random.choice(range(x.size(-1)), x.size(0))
            ).to(x.device)
        feature_idx = feature_idx_init.repeat_interleave(
                sample_size, 0
                ).unsqueeze(-1)
        missing_ = missing_.repeat_interleave(
                sample_size, 0
                )

        permutation = torch.argsort(
            torch.rand(
                sample_size*x.size(0),
                x.size(-1)),
            dim=-1
            ).to(x.device)

        arange = torch.arange(x.size(-1)).unsqueeze(0).repeat_interleave(
            permutation.size(0), 0
            ).to(x.device)
        pointer = arange <= torch.argmax(
            (permutation == feature_idx) * 1., -1
            ).view(-1,1)
        p_sorted = (-permutation).topk(
            permutation.size(-1), -1, sorted=True
            )[1]
        missing1 = torch.cat(
            [
                torch.diag(
                    pointer[:,p_sorted[:,i]]
                    ).view(-1,1) for i in range(
                        p_sorted.size(-1)
                        )
            ], -1
                        )
        missing2 = missing1.masked_fill(arange == feature_idx, False)
        missing = torch.cat([missing1 * missing_, missing2 * missing_], 0)
        x_repeat =  torch.cat(
            [
                x.repeat_interleave(sample_size, 0),
                x.repeat_interleave(sample_size, 0)
                ],
            0
            )
        prior_loc, predictive_loc,\
            predictive_scale, predictions = self.compute_loc_scale(
                x_repeat, missing
                )

        p1, p2 = predictions.split(predictions.size(0) // 2, 0)

        shapley_loc = (p1 - p2).view(orig_size, -1)
        shapley_loc1, shapley_loc2 = shapley_loc.split(
            shapley_loc.size(-1) // 2, -1
            )
        shapley_loc1 = shapley_loc1.mean(-1)
        shapley_loc2 = shapley_loc2.mean(-1)

        return shapley_loc1, shapley_loc2, feature_idx_init

    def survival_times(self, x, times):
        """
        Computes the survival at specific times. Typically, we will input
        quantiles.
        """
        with torch.no_grad():
            time_steps = len(times)
            y_ = torch.tensor(times).to(x.device)
            y = torch.cat([y_] * x.size(0), 0)
            x = x.repeat_interleave(time_steps, 0)
            m = torch.ones_like(x)
            phi, predictive_loc, predictive_scale,\
                predictions = self.compute_loc_scale(x, m)
            pc_x, logpc_x = self.simplex_net(phi)
            predictive_loc = torch.cat(
            [_.view(y.size(0),1) for _ in predictive_loc.split(y.size(0),0)],
            -1
            )
            predictive_scale = torch.cat(
            [_.view(y.size(0),1) for _ in predictive_scale.split(y.size(0),0)],
            -1
            )
            survival = torch.sum(
                pc_x * (1 - Normal(
                loc=predictive_loc,
                scale=predictive_scale
                ).cdf(
                    y.unsqueeze(-1).repeat_interleave(self.d_clusters, -1))), -1
                    )
            survival = torch.cat(
                [_.view(1, time_steps) for _ in survival.split(time_steps, 0)],
                0
                )
            survival = to_np(survival)
            survival = pd.DataFrame(survival, columns=to_np(y_))

        return survival

    def survival_horizon(self, x, start_time=0, end_time=10, time_steps=100):
        """
        Computes the survival table given instances.
        """
        with torch.no_grad():
            y_ = torch.linspace(start_time, end_time, time_steps).to(x.device)
            y = torch.cat([y_] * x.size(0), 0)
            x = x.repeat_interleave(time_steps, 0)
            m = torch.ones_like(x)
            m = self.missingness_indicator(x)
            phi, predictive_loc, predictive_scale,\
              predictions  = self.compute_loc_scale(x, m)
            pc_x, logpc_x = self.simplex_net(phi)

            predictive_loc = torch.cat(
                [_.view(y.size(0),1) for _ in predictive_loc.split(y.size(0),0)],
                -1
                )
            predictive_scale = torch.cat(
                [_.view(y.size(0),1) for _ in predictive_scale.split(y.size(0),0)],
                -1
                )
            survival = torch.sum(
                pc_x * (1 - Normal(
                loc=predictive_loc,
                scale=predictive_scale
                ).cdf(
                    y.unsqueeze(-1).repeat_interleave(self.d_clusters, -1))), -1
                    )
            survival = torch.cat(
                [_.view(1, time_steps) for _ in survival.split(time_steps, 0)],
                0
                )
            survival = to_np(survival)
            survival = pd.DataFrame(survival, columns=to_np(y_))
            return survival

if __name__ == '__main__':

    #Test config

    batch_size = 4
    d_in = 3
    d_clusters = 4
    d_hid = 200
    d_emb = 10
    n_layers = 3
    activation = 'relu'
    norm = None
    p = 0
    beta = 1

    x = torch.randn(batch_size,d_in) #covariates
    y = torch.randn(batch_size) #time to event
    c = torch.randint(low=0, high=2, size=(batch_size,)) #censoring indicator

    #Init model as self (may not be the best practice)

    self = Model(
        d_in, d_clusters, d_hid, d_emb,
        n_layers, activation, norm, p, beta
        )

    missing_ = self.missingness_indicator(x)
    sample_size = 3
    print('Items are:')
    print(self.forward(x, y, c))
