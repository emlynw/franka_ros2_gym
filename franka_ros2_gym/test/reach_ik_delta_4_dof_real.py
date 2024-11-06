import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealEnv", render_mode=render_mode, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=200)    
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
                pixels = obs["images"]["front_depth"]
                cv2.imshow("pixels", cv2.resize(pixels, display_resolution))
                cv2.waitKey(waitkey)

            if i < 15:
                action = np.array([0.2, 0.0, 0.0, 0.0, -1.0])
            elif i < 50:
                action = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
            elif i < 80:
                action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
            elif i < 100:
                action = np.array([0.0, 0.0, 0.2, 0.0, 1.0])
            elif i < 120:
                action = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
            elif i < 140:
                action = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
            else:
                action = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs['images']['front_depth'].shape)
            i+=1
        
if __name__ == "__main__":
    main()
