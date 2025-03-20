import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from wrappers import ActionRepeat, VideoRecorderReal, CustomPixelObservation, PixelFrameStack, StateFrameStack, FrankaObservation, DualCamObservation, ActionState, RotateImage, SERLObsWrapper
# from encoder_wrappers import dinov2Wrapper
import cv2
from franka_ros2_gym import envs
import numpy as np
import time
import torch
from pathlib import Path
import os
from gamepad_wrapper import GamepadIntervention

def main():
    state_res=84
    video_res=256
    action_repeat=1
    record=True
    proprio_keys = ["tcp_pose", "gripper_pos"]
    cameras = ["wrist1", "wrist2"]
    dir = os.path.dirname(__file__)
    video_dir = os.path.join(dir, 'test_vids')
    frame_stack = 2
    render_mode = "rgb_array"

    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode='rgb_array', ee_dof = 6, cameras=cameras, pos_scale=0.008, rot_scale=0.1, gripper_pause=False)
    video_dir=os.path.join(video_dir, "teleop_vids")
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps=300)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")
    env = GamepadIntervention(env)
    if record:
        for image_name in cameras:
                crop_res = env.observation_space[image_name].shape[0]
                env = VideoRecorderReal(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=20, record_every=1)
    waitkey = 1

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()

        print("press any key to start episode")
        for camera in cameras:
            cv2.imshow(camera, cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), (720, 720)))
        cv2.waitKey(0)

        while not terminated and not truncated:
            for camera in cameras:
                cv2.imshow(camera, cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)

            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']
            
            obs, reward, terminated, truncated, info = env.step(action)
        
if __name__ == "__main__":
    main()
