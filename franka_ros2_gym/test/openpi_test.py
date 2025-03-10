import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
import time
from franka_env.envs.wrappers import Quat2EulerWrapper
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)



np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, height=224, width=224, ee_dof=6, pos_scale= 0.01, rot_scale = 0.2)
    # env = Quat2EulerWrapper(env)
    env = TimeLimit(env, max_episode_steps=50)    
    waitkey = 1

    instruction = "pick the red straberry"
    obs, info = env.reset()
    droid_obs = {
         "wrist_image": obs["images"]["wrist1"],
         "cartesian_position": obs['state']['tcp_pose'][0:3],
         "gripper_position": obs['state']['gripper_pos']
    }

    initial_pose = obs['state']['tcp_pose']
    for i in range(500):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])


        obs, reward, terminated, truncated, info = env.step(action)
        droid_obs = {
            "wrist_image": obs["images"]["wrist1"],
            "cartesian_position": obs['state']['tcp_pose'][0:3],
            "gripper_position": obs['state']['gripper_pos']
        }


        if render_mode == "rgb_array":
                pixels = obs["images"]["wrist1"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
    final_pose = obs['state']['tcp_pose']
        
        
if __name__ == "__main__":
    main()
