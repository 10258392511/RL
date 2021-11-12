import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from Piazza
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        observation_tensor = torch.FloatTensor(observation, device=ptu.device)
        action_dist = self.forward(observation_tensor)
        action = action_dist.sample()

        return action.detach().cpu().numpy()
        # return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from Piazza
        if self.discrete:
            action_distribution = torch.distributions.Categorical(logits=self.logits_na(observation))
        else:
            action_mean = self.mean_net(observation)
            # print(f"inside MLPolicy: .forward(.): obs: {observation.shape}, action_mean: {action_mean.shape}, log_std: {self.logstd.shape}")
            action_distribution = torch.distributions.Normal(action_mean, self.logstd.exp().unsqueeze(0))

        return action_distribution


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        adv_n = ptu.from_numpy(adv_n) if adv_n is not None else None

        # # (1000, 4), (1000,), (1000,)
        # print(f"inside MLPoicy: observations: {observations.shape}, actions: {actions.shape}, adv_n: {adv_n.shape if adv_n is not None else None}")
        # print("-" * 40)
        action_dist = self.forward(observations)
        log_prob = action_dist.log_prob(actions)
        if not self.discrete:
            # discrete: categorical, direct log_prob, (B,)
            # continous: sum over all dims, (B, ac_dim) -> (B,)
            log_prob = log_prob.sum(dim=-1)
        loss = -(action_dist.log_prob(actions) * adv_n).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
