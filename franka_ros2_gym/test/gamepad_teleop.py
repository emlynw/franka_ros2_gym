import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, pos_scale = 0.01, rot_scale=0.2, cameras=['wrist2'], randomize_domain=False, ee_dof=6)
    env = GamepadIntervention(env)
    env = TimeLimit(env, max_episode_steps=50000)    
    waitkey = 10
    resize_resolution = (480, 480)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        rotate = True
        
        while not terminated and not truncated:
            step_start_time = time.time()
            wrist2 = obs["images"]["wrist2"]
            cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR), resize_resolution))
            # wrist1 = cv2.rotate(obs['images']['wrist1'], cv2.ROTATE_180)
            # cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))
            cv2.waitKey(waitkey)
            
    
            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs['state']['tcp_pose'])
            step_time = time.time()-step_start_time
            if step_time < 0.05:
                time.sleep(0.05 - step_time)
            i+=1
        
if __name__ == "__main__":
    main()
