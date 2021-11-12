import mujoco_py
import gym
from pyvirtualdisplay import Display
from cs285.infrastructure.colab_utils import wrap_env, show_video


if __name__ == "__main__":
    # # Medium test
    # env = gym.make("FetchReach-v1")
    # env.reset()

    # for _ in range(300):
    #     env.render()
    #     env.step(env.action_space.sample())
    # env.close()

    # HW1 test
    env = gym.make("Ant-v2")
    obs = env.reset()
    print(f"obs: {obs}, {type(obs)}")
    for i in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        if i == 10:
            print(f"obs: {obs}\nrew: {rew}\ndone: {done}\ninfo:{info}")
    