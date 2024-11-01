# https://github.com/rail-berkeley/serl/blob/e2065d673131af6699aa899a78159859bd17c135/franka_sim/franka_sim/envs/panda_pick_gym_env.py
import numpy as np
import os

from gymnasium import utils
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from pathlib import Path
from scipy.spatial.transform import Rotation

import rclpy
from rclpy import qos
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError

class ReachIKDeltaEnv(gym.env, utils.EzPickle):

    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs=True,
        ee_dof = 6, # 3 for position, 3 for orientation
        width=480,
        height=480,
        render_mode="rgb_array",
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)
        self.image_obs = image_obs
        self.ee_dof = ee_dof
        self.render_mode = render_mode
        self.width = width
        self.height = height

        state_space = Dict(
            {
                "panda/tcp_pos": Box(np.array([0.28, -0.5, 0.01]), np.array([0.75, 0.5, 0.55]), shape=(3,), dtype=np.float32),
                "panda/tcp_orientation": Box(-1, 1, shape=(4,), dtype=np.float32),  # Quaternion
                "panda/gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
                "panda/gripper_vec": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.observation_space = Dict({"state": state_space})
        if image_obs:
            self.observation_space["images"] = Dict(
                {
                    "wrist": Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
                    "front": Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
                }
            )

        self.camera_id = (0, 1)
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )

        self.setup()
        self.ros2_setup()

    def setup(self):
        # self._PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
        self._PANDA_HOME = np.array([-0.00171672, -0.786471, -0.00122413, -2.36062, 0.00499334, 1.56444, 0.772088], dtype=np.float32)
        self._PANDA_XYZ = np.array([0.3, 0, 0.5], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.28, -0.35, 0.01], [0.75, 0.35, 0.55]], dtype=np.float32)
        self._ROTATION_BOUNDS= np.array([[-np.pi/4, -np.pi/4, -np.pi/2], [np.pi/4, np.pi/4, np.pi/2]], dtype=np.float32)

        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id

        # Define action scaling factors
        self.pos_scale = 0.1  # Maximum position change (in meters)
        self.rot_scale = 0.05  # Maximum rotation change (in radians)
        
        self.reset_arm_and_gripper()

        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_grasp_time = 0.0
        self.prev_grasp = -1.0
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }

        # Add this line to set the initial orientation
        self.initial_orientation = [0, 1, 0, 0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

    def ros2_setup(self):
        self.node = rclpy.create_node('robot_env_node')
        # Subscribers
        self.state_sub = self.node.create_subscription(FrankaState, '/franka_robot_state_broadcaster/robot_state', 
                                                  self.state_callback, qos_profile=qos.qos_profile_sensor_data)
        self.gripper_sub = self.node.create_subscription(JointState, '/panda_gripper/joint_states', self.gripper_callback,
                                                    qos_profile=qos.qos_profile_sensor_data)
        self.image_sub = self.node.create_subscription(Image, '/color/image_raw', 
                                                  self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        # Publishers
        self.goal_pose_pub = self.create_publisher(Pose, '/franka/goal_pose', 10)
        self.gripper_pub = self.create_publisher(Float32, '/franka/gripper', 10)

    def state_callback(self, data):
        return 
    
    def gripper_callback(self, data):
        return

    def image_callback(self, data):
        return

    def domain_randomization(self):
        # Move robot
        ee_noise = np.random.uniform(low=[0.0,-0.2,-0.4], high=[0.12, 0.2, 0.1], size=3)
        self.data.mocap_pos[0] = self._PANDA_XYZ + ee_noise
        
    def reset_arm_and_gripper(self):
        return        


    def reset(self):
        info = {}
        self.reset_arm_and_gripper()
        if self.randomize_domain:
            self.domain_randomization()

        self.gripper_vec = self.gripper_dict["open"]
        self.data.ctrl[self._gripper_ctrl_id] = 255
        self.prev_grasp_time = 0.0
        self.prev_gripper_state = 0 # 0 for open, 1 for closed
        self.gripper_state = 0
        self.gripper_blocked = False

        return self._get_obs(), info

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Scale actions
        if self.ee_dof == 3:
            x, y, z, grasp = action
        elif self.ee_dof == 4:
            x, y, z, yaw, grasp = action
            roll, pitch = 0, 0
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        elif self.ee_dof == 6:
            x, y, z, roll, pitch, yaw, grasp = action
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        dpos = np.array([x, y, z]) * self.pos_scale
        
        # Apply position change
        pos = self.data.sensor("pinch_pos").data
        npos = np.clip(pos + dpos, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        if self.ee_dof > 3:
            # Orientation changes, ZYX because of mujoco quaternions?
            current_quat = self.data.sensor("pinch_quat").data
            current_rotation = Rotation.from_quat(current_quat)
            # Convert the action rotation to a Rotation object
            action_rotation = Rotation.from_euler('zyx', drot)
            # Apply the action rotation
            new_rotation = action_rotation * current_rotation
            # Calculate the new relative rotation
            new_relative_rotation = self.initial_rotation.inv() * new_rotation
            # Convert to euler angles and clip
            relative_euler = new_relative_rotation.as_euler('zyx')
            clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
            # Convert back to rotation and apply to initial orientation
            clipped_rotation = Rotation.from_euler('zyx', clipped_euler)
            final_rotation = self.initial_rotation * clipped_rotation
            # Set the final orientation
            self.data.mocap_quat[0] = final_rotation.as_quat()

        # Handle grasping
        grasp = int(grasp>0)
        if self.data.time - self.prev_grasp_time < 0.5:
            self.gripper_blocked = True
            grasp = self.prev_grasp
        else:
            self.gripper_blocked = False
            if grasp == 0 and self.gripper_state == 0:
                self.gripper_vec = self.gripper_dict["open"]
            elif grasp == 1 and self.gripper_state == 1:
                self.gripper_vec = self.gripper_dict["closed"]
            elif grasp == 0 and self.gripper_state == 1:
                self.data.ctrl[self._gripper_ctrl_id] = 255
                self.gripper_state = 0
                self.gripper_vec = self.gripper_dict["opening"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp
            elif grasp == 1 and self.gripper_state == 0:
                self.data.ctrl[self._gripper_ctrl_id] = 0
                self.gripper_state = 1
                self.gripper_vec = self.gripper_dict["closing"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp


        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._compute_reward(action)
        self.prev_gripper_state = self.gripper_state

        return obs, reward, False, False, info 
    
    def render(self):

        return 

    def _get_obs(self):
        obs = {"state": {}}

        # Populate state observations
        obs["state"]["panda/tcp_pos"] = self.data.sensor("pinch_pos").data.astype(np.float32)
        obs["state"]["panda/tcp_orientation"] = self.data.sensor("pinch_quat").data.astype(np.float32)
        obs["state"]["panda/tcp_vel"] = self.data.sensor("pinch_vel").data.astype(np.float32)
        obs["state"]["panda/gripper_pos"] = (25*2*np.array([self.data.qpos[8]], dtype=np.float32)-1)
        obs["state"]["panda/gripper_vec"] = self.gripper_vec

        if not self.image_obs:
            obs["state"]["block_pos"] = self.data.sensor("block_pos").data.astype(np.float32)
        if self.image_obs:
            obs["images"] = {}
            obs["images"]["wrist"], obs["images"]["front"] = self.render()

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _compute_reward(self, action):
        tcp_pos = self.data.sensor("pinch_pos").data
        # Reward for staying near center_pos
        r_move = 1 - np.tanh(5 * np.linalg.norm(tcp_pos[:2] - self.center_pos[:2]))
        # Smoothness reward
        action_diff = np.linalg.norm(action[:-1] - self.prev_action[:-1])/np.sqrt((len(action)-1)) # Divide by sqrt of action dimension to make it scale-invariant
        r_smooth = 1 - np.tanh(5 * action_diff)
        self.prev_action = action
        # Reward for not interrupting grasp
        if self.gripper_blocked and self.gripper_state != self.prev_gripper_state:
            r_grasp = 0
        else:
            r_grasp = 1

        rewards = {'r_smooth': r_smooth, 'r_grasp': r_grasp, 'r_move': r_move}
        reward_scales = {'r_smooth': 0.5, 'r_grasp': 0.5, 'r_move': 1.0}

        rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
        reward = np.clip(sum(rewards.values()), -1e4, 1e4)
        
        info = rewards
        return reward, info