import numpy as np

def _discounted_return(self, rewards):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """

    # TODO: create list_of_discounted_returns
    # Hint: note that all entries of this output are equivalent
        # because each sum is from 0 to T (and doesnt involve t)
    cur_reward_acc = 0
    gamma_acc = 1
    for reward in rewards:
        cur_reward_acc += gamma_acc * reward
        gamma_acc *= gamma
    
    list_of_discounted_returns = [cur_reward_acc for _ in range(len(rewards))]

    return list_of_discounted_returns


def _discounted_cumsum(gamma, rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """

    # TODO: create `list_of_discounted_returns`
    # HINT1: note that each entry of the output should now be unique,
        # because the summation happens over [t, T] instead of [0, T]
    # HINT2: it is possible to write a vectorized solution, but a solution
        # using a for loop is also fine
    def weighted_sum(rewards, gamma):
        weights = gamma ** np.arange(len(rewards))
        return weights @ np.array(rewards)
    
    list_of_discounted_cumsums = []
    for i in range(len(rewards)):
        list_of_discounted_cumsums.append(weighted_sum(rewards[i:], gamma))

    return list_of_discounted_cumsums

if __name__ == "__main__":
    gamma = 0.9
    rewards = [1, 2, 3]
    out = _discounted_return(gamma, rewards)
    print(out)

    out = _discounted_cumsum(gamma, rewards)
    print(out)
