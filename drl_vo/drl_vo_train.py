#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to train the DRL-VO policy using the PPO algorithm.
#------------------------------------------------------------------------------

import os
import numpy as np
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from custom_cnn_full import *
import env_gym


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.val_results = []
        self.last_success_times = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
          os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # evaluate model
          print('start to evaluate model')
          success_times = 0
          collision_times = 0
          navigation_time = []
          eps = 100
          for i in range(eps):
            obs = env_val.reset()
            done = False
            ep_step = 0
            while not done:
              action, _states = model.predict(obs)
              obs, reward, done, info = env_val.step(action)
              ep_step = ep_step + 1
              if done:
                if info['arrival']:
                  success_times = success_times + 1
                  navigation_time.append(ep_step * 0.2)
                elif info['collision']:
                  collision_times = collision_times + 1
          ave_navigation_time = 25.0
          if len(navigation_time) > 0:
            ave_navigation_time = sum(navigation_time) / len(navigation_time)
          self.val_results.append(np.array([self.n_calls, success_times / eps, collision_times / eps, ave_navigation_time]))

          print('training steps, success rate, collision, navigation time: ', self.n_calls, success_times, collision_times, ave_navigation_time)
          np.savetxt(log_dir + 'val_result.txt', np.array(self.val_results))
          # New best model, you could save the agent here
          if success_times > self.last_success_times:
              self.last_success_times = success_times
              # Example for saving best model
              print("Saving new best model to {}".format(self.save_path))
              self.model.save(self.save_path)
                  
        # save model every 100000 timesteps:
        if self.n_calls % (20000) == 0:
          # Retrieve training reward
          print('save model')
          path = self.save_path + '_model' + str(self.n_calls)
          self.model.save(path)
	  
        return True



# Create log dir
log_dir = './runs/'
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('drl-nav-v0')
env.configure()
obs = env.reset()

env_val = gym.make('drl-nav-v0')
env_val.configure()

# policy parameters:
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(pi=[256], vf=[128])],
)

# raw training:
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, device='cuda:2', learning_rate=1e-3, verbose=2, \
  tensorboard_log=log_dir, n_steps=512, n_epochs=10, batch_size=128) #, gamma=0.96, ent_coef=0.1, vf_coef=0.4) 

# continue training:
# kwargs = {'tensorboard_log':log_dir, 'verbose':2, 'n_epochs':10, 'n_steps':512, 'batch_size':128,'learning_rate':5e-5}
# model_file = rospy.get_param('~model_file', "./model/drl_pre_train.zip")
# model = PPO.load(model_file, env=env, **kwargs)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
print('start to traing the model')
model.learn(total_timesteps=1000000, log_interval=5, tb_log_name='drl_vo_policy', callback=callback, reset_num_timesteps=True)

# Saving final model
model.save("drl_vo_model")
print("Training finished.")


