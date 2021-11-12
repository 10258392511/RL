import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]  # (1, H, W, C * L) or (1, obs_dim)
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        actions_out = self.critic.qa_values(observation)  # (B, action_dim)
        # print(f"inside ArgMaxPolicy: actions_out: {actions_out.shape}")
        actions = actions_out.argmax(axis=1)

        return actions.squeeze()
