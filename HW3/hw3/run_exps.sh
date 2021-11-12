#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo "requires 1 argument"
    exit 1
fi

if [[ $1 -eq 1 ]]
then
    echo "running test 1"
    # python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1

    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3

    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2
    python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3

    # python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1
    # python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2
    # python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3


elif [[ $1 -eq 2 ]]
then
    echo "running test 2"
    # python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1

    python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1
    python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100
    python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10

elif [[ $1 -eq 3 ]]
then
    echo "extra tests or debug"
    python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
fi
