from gym.envs.registration import register

# Register drl_nav env 
register(
  id='drl-nav-v0',
  entry_point='env_gym.envs:DRLNavEnv'
  )


