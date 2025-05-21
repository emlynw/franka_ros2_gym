from gymnasium.envs.registration import register

register( id="franka_ros2_gym/ReachIKDeltaRealEnv-v0", entry_point="franka_ros2_gym.envs:ReachIKDeltaRealEnv" , max_episode_steps=1000)
register( id="franka_ros2_gym/ReachIKDeltaRealStrawbEnv-v0", entry_point="franka_ros2_gym.envs:ReachIKDeltaRealStrawbEnv" , max_episode_steps=1000)
register( id="franka_ros2_gym/ReachIKAbsRealStrawbEnv-v0", entry_point="franka_ros2_gym.envs:ReachIKAbsRealStrawbEnv" , max_episode_steps=1000)
