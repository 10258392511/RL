import numpy as np
import sys

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]  ## assume: (N_obs,)

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]  ## assume: dynamic range of action space
        
        random_action_sequences = self.low + (self.high - self.low) * np.random.rand(num_sequences, horizon, self.ac_dim)

        return random_action_sequences

    def get_action(self, obs):

        ## assume: obs: (B, N_obs)
        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon)

        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        predicted_rewards = np.mean(
            predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N  ## ens = number of ensembles, N: number of action seqs
        ## each ensemble runs on N sequences, and for each sequence, compute the average

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[predicted_rewards.argmax(axis=-1)]  # TODO (Q2)  ## int; (N, H, N_ac) -> (H, N_ac)
        action_to_take = best_action_sequence[0]  # TODO (Q2)  ## (H, N_ac) -> (N_ac,)
        return action_to_take[None]  # Unsqueeze the first index  ## (1, N_ac)

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        ## assume: obs: (N_obs,), candidate_action_seq: (N, H, N_ac)
        sum_of_rewards = None  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        ## at time t: apply (N, N_ac) actions to (N, N_obs) observations, so they're independent (see notes in script)
        N, H, _ = candidate_action_sequences.shape  ## N: 1000, H: 10
        # print(f"N: {N}, H: {H}")
        rewards = np.zeros((N, H))
        obs_predictions = np.zeros((N, H, self.ob_dim))
        obs_predictions[:, 0, :] = np.tile(obs[None, :], (N, 1))  ## (N_obs,) -> (1, N_obs) -> (N, N_obs) for h = 0:, store original obs
        for t in range(H):
            ## we have candidate actions, so there's no need to predict actions
            ## (N, N_obs), (N, N_ac) -> (N,), ((N,)), batch version
            rewards[:, t], _ = self.env.get_reward(obs_predictions[:, t, :], candidate_action_sequences[:, t - 1, :])
            # print(f"actions: {actions[0].shape}, {actions[1].shape}")
            # sys.exit()
            if t < H - 1:
                ## (N, N_obs), (N, N_ac) -> (N, N_obs)
                obs_predictions[:, t + 1, :] = model.get_prediction(obs_predictions[:, t, :], candidate_action_sequences[:, t, :], self.data_statistics) 
            
        sum_of_rewards = rewards.mean(axis=1)

        return sum_of_rewards
