import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
np.set_printoptions(suppress=True)

# Global variables to capture mouse movement
mouse_x, mouse_y = 0, 0  # Track mouse position

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, cameras=['wrist1', 'wrist2'], ee_dof=4)
    env = TimeLimit(env, max_episode_steps=1000)    
    waitkey = 10
    resize_resolution = (720, 720)

    # Define the range for absolute movement control
    max_speed = 0.1  # Maximum speed in any direction
    rot_speed = 0.2  # Maximum rotation speed

    # Set up mouse callback
    cv2.namedWindow("wrist2")
    cv2.setMouseCallback("wrist2", mouse_callback)
    
    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        while not terminated and not truncated:
            # Display the environment
            if render_mode == "rgb_array":
                cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(obs['images']['wrist2'], cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(obs["images"]["wrist1"], cv2.COLOR_RGB2BGR), resize_resolution))
            
            # Calculate movement based on absolute mouse position within window
            move_left_right = ((mouse_x / resize_resolution[0]) * 2 - 1) * max_speed
            move_up_down = -((mouse_y / resize_resolution[1]) * 2 - 1) * max_speed

            # Define movement actions for W and S keys (forward/backward)
            key = cv2.waitKey(waitkey) & 0xFF
            move_action = np.array([0, move_left_right, move_up_down, 0.0, 0.0])  # Default move

            if key == ord('w'):
                move_action[0] = max_speed  # Forward
            elif key == ord('s'):
                move_action[0] = -max_speed   # Backward
            elif key == ord('a'):
                move_action[3] = -rot_speed
            elif key == ord('d'):
                move_action[3] = rot_speed

            # Toggle gripper state with spacebar
            if key == ord(' '):
                move_action[-1] = 1.0
            elif key == ord('c'):
                move_action[-1] = -1.0

            # Perform the action in the environment
            obs, reward, terminated, truncated, info = env.step(move_action)

            # Reset environment on 'R' key press
            if key == ord('r'):
                print("Resetting environment...")
                obs, info = env.reset()  # Reset the environment
                continue  # Start the loop again after reset

            # Exit on 'ESC' key
            if key == 27:  # ESC key
                print("Exiting...")
                env.close()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
