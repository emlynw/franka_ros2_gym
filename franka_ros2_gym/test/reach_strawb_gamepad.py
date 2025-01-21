import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
import pygame

np.set_printoptions(suppress=True)

def apply_dead_zone(value, DEAD_ZONE=0.15):
    if abs(value) < DEAD_ZONE:
        return 0.0
    elif value > 0:
        return (value - DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the positive range
    else:
        return (value + DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the negative range

def main():
    render_mode = "rgb_array"
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", render_mode=render_mode, cameras=['wrist1', 'wrist2'], randomize_domain=False, ee_dof=6)
    env = TimeLimit(env, max_episode_steps=1000)
    resize_resolution = (480, 480)
    waitkey = 10

    # Initialize pygame for Xbox controller input
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Please connect a controller.")
        return  # Exit if no joystick is connected
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Using controller: {joystick.get_name()}")

    max_speed = 0.3  # Maximum speed in any direction
    max_z_speed = 0.1  # Maximum speed in the z direction
    rot_speed = 1.0  # Maximum rotational speed
    # Dead zone threshold
    DEAD_ZONE = 0.15 # Adjust as needed
    i=0

    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        i=0

        while not terminated and not truncated:
            i+=1
            # Display the environment
            if render_mode == "rgb_array":
                wrist2 = obs["images"]["wrist2"]
                cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR), resize_resolution))
                wrist1 = cv2.rotate(obs['images']['wrist1'], cv2.ROTATE_180)
                cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))

            pygame.event.pump()  # Process events

            # Get joystick axes for movement and rotation
            left_stick_x = apply_dead_zone(joystick.get_axis(0)) # Left stick horizontal (left/right)
            left_stick_y = apply_dead_zone(joystick.get_axis(1)) # Left stick vertical (up/down)
            right_stick_x = apply_dead_zone(joystick.get_axis(3)) # Right stick horizontal
            right_stick_y = apply_dead_zone(joystick.get_axis(4)) # Right stick vertical
            trigger_l = apply_dead_zone(joystick.get_axis(2)) # Left trigger (gripper open/close)
            trigger_r = apply_dead_zone(joystick.get_axis(5)) # Right trigger (gripper close/open)

            # Compute actions
            move_forward_backward = -left_stick_y * max_speed  # Forward/backward
            move_left_right = left_stick_x * max_speed  # Left/right
            move_up_down = (trigger_r - trigger_l) * max_z_speed  # Up/down based on triggers

            # Check if left bumper (LB) is pressed
            is_roll_mode = joystick.get_button(4)  # Left bumper (button 4)

            # Determine rotation inputs
            if is_roll_mode:
                roll = right_stick_x * rot_speed  # Right stick horizontal controls roll
                yaw = 0.0 # Disable yaw while rolling
            else:
                roll = 0.0
                yaw = -right_stick_x * rot_speed # Right stick horizontal controls yaw

            pitch = right_stick_y * rot_speed  # Right stick vertical always controls pitch

            # Create the action array
            move_action = np.array([move_forward_backward, move_left_right, move_up_down, roll, pitch, yaw, 0.0])

            # Gripper control with A button (button 0) for toggling
            if joystick.get_button(0):  # A button toggles gripper
                move_action[-1] = 1.0
            # If x button is pressed, close the gripper
            if joystick.get_button(2):  # X button closes gripper
                move_action[-1] = -1.0

            # Perform the action in the environment
            obs, reward, terminated, truncated, info = env.step(move_action)
            print(i)

            # Check for reset or exit
            key = cv2.waitKey(waitkey) & 0xFF
            if key == ord('r'):  # Reset the environment
                print("Resetting environment...")
                obs, info = env.reset()
                i=0
                continue  # Restart the loop after reset
            if key == 27:  # ESC key to exit
                print("Exiting...")
                env.close()
                cv2.destroyAllWindows()
                pygame.quit()
                return

if __name__ == "__main__":
    main()
