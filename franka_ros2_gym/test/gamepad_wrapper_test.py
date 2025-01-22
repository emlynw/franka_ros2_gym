import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, pos_scale = 0.2, rot_scale=1.0, cameras=['wrist1', 'wrist2'], randomize_domain=False, ee_dof=6)
    env = GamepadIntervention(env)
    env = TimeLimit(env, max_episode_steps=500)    
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
            wrist2 = obs["images"]["wrist2"]
            cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR), resize_resolution))
            wrist1 = cv2.rotate(obs['images']['wrist1'], cv2.ROTATE_180)
            cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))
            cv2.waitKey(waitkey)
            
    
            action = np.zeros_like(env.action_space.sample())
            print(f"action: {action}")
            if "intervene_action" in info:
                print(f"i action: {info['intervene_action']}")
                action = info['intervene_action']

            print(F"action: {action}")

            
            obs, reward, terminated, truncated, info = env.step(action)
            i+=1
        
if __name__ == "__main__":
    main()
