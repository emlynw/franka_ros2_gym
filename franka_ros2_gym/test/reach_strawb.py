import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, ee_dof=6, cameras=["wrist2"])
    env = TimeLimit(env, max_episode_steps=40)    
    waitkey = 1
    display_resolution = (720, 720)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        while not terminated and not truncated:
            if render_mode == "rgb_array":
                pixels = obs["images"]["wrist2"]
                cv2.imshow("pixels", cv2.resize(pixels, display_resolution))
                cv2.waitKey(waitkey)

            if i < 20:
                action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"cartesian pos: {obs['state']['panda/tcp_pos']}")
            print(f"cartesian ori: {obs['state']['panda/tcp_orientation']}")
            i+=1
        
if __name__ == "__main__":
    main()
