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

def load_snapshot():
        snapshot =  Path('/home/emlyn/rl_franka/drm_backbones/test_snapshots/2025_02_18_225201/1530000/snapshot.pt')
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload

def main():
    state_res=84
    video_res=480
    action_repeat=1
    record=True
    proprio_keys = ["tcp_pose", "gripper_pos"]
    cameras = ["wrist1", "wrist2"]
    dir = os.path.dirname(__file__)
    video_dir = os.path.join(dir, 'test_vids')
    frame_stack = 2
    render_mode = "rgb_array"

    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode='rgb_array', ee_dof = 6, cameras=cameras, rot_scale=0.1)
    video_dir=os.path.join(video_dir, "inference_vids")
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps=300)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")
    if record:
        for image_name in cameras:
                crop_res = env.observation_space[image_name].shape[0]
                env = VideoRecorderReal(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=20, record_every=2)
    env = ActionState(env)
    env = CustomPixelObservation(env, pixel_key="wrist1", crop_resolution=video_res, resize_resolution=state_res)
    env = CustomPixelObservation(env, pixel_key="wrist2", crop_resolution=video_res, resize_resolution=state_res)
    env = PixelFrameStack(env, frame_stack, stack_key="wrist1")
    env = PixelFrameStack(env, frame_stack, stack_key="wrist2")
    env = StateFrameStack(env, frame_stack)
    waitkey = 1

    payload = load_snapshot()
    print(payload)
    agent = payload['agent']

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()

        print("press any key to start episode")
        for camera in cameras:
            cv2.imshow(camera, cv2.resize(cv2.cvtColor(obs[camera][0:3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR), (720, 720)))
        cv2.waitKey(0)

        while not terminated and not truncated:
            for camera in cameras:
                cv2.imshow(camera, cv2.resize(cv2.cvtColor(obs[camera][0:3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)

            wrist1 = obs['wrist1'].astype(np.float32)
            wrist2 = obs['wrist2'].astype(np.float32)
            state = obs['state'].astype(np.float32)
            act_start_time = time.time()
            with torch.no_grad():
                    action = agent.act(wrist1, wrist2, state, payload['_global_step'], eval_mode=True)
            print(f"act time: {time.time()-act_start_time}")
            obs, reward, terminated, truncated, info = env.step(action)
            i+=1
        
if __name__ == "__main__":
    main()
