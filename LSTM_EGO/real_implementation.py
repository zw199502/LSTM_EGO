import argparse
import numpy as np
import random
import os
import torch
import time
from crowd_real_implementation import CrowdSim

from algos import TD3
from utils import memory
from info import *

import rospy
import time

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=100, save_directory=None):
    policy.eval_mode()
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    for i in range(eval_episodes):
        lidar_state, position_state  = eval_env.reset()
        # eval_env.render()
        # time.sleep(0.03)
        done = False
        hidden = None
        ep_step = 0
        while not done:
            with torch.no_grad():
                action, hidden = policy.select_action(lidar_state, position_state, hidden)
            
            lidar_state, position_state, reward, done, info = eval_env.step(action)
            # eval_env.render()
            # time.sleep(0.03)
            avg_reward += reward
            ep_step = ep_step + 1
        if save_directory is not None:
            file_name = save_directory + '/eval_' + str(time.time()) + '.npz'
            if i < 10:
                np.savez_compressed(file_name, **eval_env.log_env)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
            print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
            print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
            print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
        else:
            raise ValueError('Invalid end signal from environment')

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit

    policy.train_mode()
    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    # To Do, revise DDPG according to TD3
    parser.add_argument("--policy", default="TD3")
    # device
    parser.add_argument("--device", type=str, default='cuda:0')
    # OpenAI gym environment name
    parser.add_argument("--env", default="crowd_real")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.25)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e5, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=512, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True, action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    # parser.add_argument("--load_model", type=str, default="")
    # 500 tests, 0.89 success rate
    parser.add_argument("--load_model", type=str, default="/models/step_150000success_97")
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true", default=True)
    # environment settings
    parser.add_argument("--only_dynamic", action="store_true", default=True)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--lidar_dim", type=int, default=1800)
    parser.add_argument("--lidar_feature_dim", type=int, default=50)
    parser.add_argument("--goal_position_dim", type=int, default=2)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=0.27)
    parser.add_argument("--laser_max_range", type=float, default=6.0)
    parser.add_argument("--square_width", type=float, default=10.0)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    

    file_prefix = '/home/zw/expand_disk/ubuntu/RNN_RL/' + args.policy + '/' + args.env + '/seed_' + str(args.seed) + '_only_dynamic'

    eval_env = CrowdSim(args)

    # Set seeds
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    lidar_state_dim = args.lidar_dim
    position_state_dim = args.goal_position_dim
    lidar_feature_dim = args.lidar_feature_dim
    action_dim = args.action_dim
    max_action = 1.0


    kwargs = {
        "lidar_state_dim": lidar_state_dim,
        "position_state_dim": position_state_dim,
        "lidar_feature_dim": lidar_feature_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
    }

    
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["device"] = args.device
    policy = TD3.TD3(**kwargs)


    policy.load(file_prefix + args.load_model)

  
    # success_rate, collision_rate, avg_nav_time = eval_policy(policy, eval_env, eval_episodes=1, save_directory=file_prefix + '/evaluation_episodes')
    # print('success_rate, collision_rate, avg_nav_time')
    # print(success_rate, collision_rate, avg_nav_time)



if __name__ == "__main__":
    rospy.init_node('navigation', anonymous=True) #make node 
    main()
    rospy.spin()
