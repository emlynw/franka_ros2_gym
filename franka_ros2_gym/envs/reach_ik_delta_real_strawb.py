import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict

import rclpy
from rclpy.node import Node
from rclpy import qos
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation
import time
import threading

class ReachIKDeltaRealStrawbEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        image_obs=True,
        ee_dof=6,  # 3 for position only, 4 for position+yaw
        width=480,
        height=480,
        pos_scale=0.2,
        rot_scale=0.2,
        control_dt=0.1,
        cameras=["wrist1", "wrist2", "front"],
        depth = False,
        **kwargs
    ):
        super().__init__()

        # Environment parameters
        self.image_obs = image_obs
        self.ee_dof = ee_dof
        self.width = width
        self.height = height
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.control_dt = control_dt
        self.cameras = cameras
        self.depth = depth
        
        # Action and observation spaces
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32
        )
        
        state_space = Dict({
            "panda/tcp_pos": Box(
                np.array([0.28, -0.35, 0.25]), 
                np.array([0.8, 0.35, 0.8]), 
                shape=(3,), 
                dtype=np.float32
            ),
            "panda/tcp_orientation": Box(-1, 1, shape=(4,), dtype=np.float32),
            "panda/tcp_vel": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "panda/gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
            "panda/gripper_vec": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        })
        
        self.observation_space = Dict({"state": state_space})
        if image_obs:
            self.observation_space["images"] = Dict()
            for camera in self.cameras:
                self.observation_space["images"][camera] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )
                if self.depth:
                    self.observation_space["images"][f"{camera}_depth"] = Box(
                        0, 65535, shape=(self.height, self.width), dtype=np.uint16
                    )

        # Control parameters
        self._CARTESIAN_BOUNDS = np.array([[0.28, -0.35, 0.25], [0.8, 0.35, 0.8]], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([[-np.pi/4, -np.pi/2, -np.pi/2],[np.pi/4, np.pi/2, np.pi/2]], dtype=np.float32)
        
        # Initial poses
        self.initial_position = np.array([0.4, 0.0, 0.7], dtype=np.float32)
        self.initial_orientation = [0.725, 0.0, 0.688, 0.0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)
        
        # ROS setup
        self.required_attributes = ["rot_mat", "x", "y", "z", "gripper_width"] + [
            camera for camera in self.cameras
        ] + [f"{camera}_depth" for camera in self.cameras if self.depth]
        for attr in self.required_attributes:
            setattr(self, attr, None)

        # Initialize attributes to None
        for attr in self.required_attributes:
            setattr(self, attr, None)
        
        # Gripper state tracking
        self.prev_grasp_time = 0.0
        self.prev_grasp = -1.0
        # self.gripper_dict = {
        #     "open": np.array([1, 0, 0, 0], dtype=np.float32),
        #     "closed": np.array([0, 1, 0, 0], dtype=np.float32),
        #     "opening": np.array([0, 0, 1, 0], dtype=np.float32),
        #     "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        # }
        # self.gripper_dict = {
        #     "moving": np.array([1, 0], dtype=np.float32),
        #     "grasping": np.array([0, 1], dtype=np.float32),
        # }
        self.gripper_dict = {
            "stopped": np.array([1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 1], dtype=np.float32),
        }
        self.gripper_vec = self.gripper_dict["stopped"]
        self.grasp = -1.0
        self.gripper_blocked = False
        self.ros_setup()
        self.last_step_time = time.time()

    def ros_setup(self):
        # Initialize ROS node
        rclpy.init()
        self.node = rclpy.create_node('reach_ik_delta_real')
        
        # ROS Publishers
        self.goal_pose_pub = self.node.create_publisher(Pose, '/franka/goal_pose', 10)
        self.gripper_pub = self.node.create_publisher(Float32, '/franka/gripper', 10)
        
        # ROS Subscribers
        custom_qos_profile = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.BEST_EFFORT,
            durability=qos.DurabilityPolicy.VOLATILE,
            history=qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
     
        self.state_sub = self.node.create_subscription(
            FrankaState, 
            '/franka_robot_state_broadcaster/robot_state',
            self.state_callback,
            qos_profile=custom_qos_profile,
            callback_group=self.callback_group
        )
        self.vel_sub = self.node.create_subscription(
            Twist,
            '/franka/cartesian_speed',
            self.vel_callback,
            qos_profile=custom_qos_profile,
            callback_group=self.callback_group
        )

        self.gripper_sub = self.node.create_subscription(
            JointState,
            '/panda_gripper/joint_states',
            self.gripper_callback,
            qos_profile=custom_qos_profile,
            callback_group=self.callback_group
        )

        self.bridge = CvBridge()
        for camera in self.cameras:
            setattr(self, f"{camera}_sub", self.node.create_subscription(
                Image, f'/{camera}/color/image_raw',
                getattr(self, f"{camera}_callback"), qos_profile=custom_qos_profile, callback_group=self.callback_group
            ))
            if self.depth:
                setattr(self, f"{camera}_depth_sub", self.node.create_subscription(
                    Image, f'/{camera}/aligned_depth_to_color/image_raw',
                    getattr(self, f"{camera}_depth_callback"), qos_profile=custom_qos_profile, callback_group=self.callback_group
                ))

        print("ROS setup complete")

    def are_attributes_initialized(self):
        """Check if all required attributes are non-None."""
        return all(getattr(self, attr) is not None for attr in self.required_attributes)
   
    def state_callback(self, data):
        self.rot_mat = np.array([
            [data.o_t_ee[0], data.o_t_ee[4], data.o_t_ee[8]],
            [data.o_t_ee[1], data.o_t_ee[5], data.o_t_ee[9]],
            [data.o_t_ee[2], data.o_t_ee[6], data.o_t_ee[10]]
        ])
        self.x = np.float32(data.o_t_ee[12])
        self.y = np.float32(data.o_t_ee[13])
        self.z = np.float32(data.o_t_ee[14])

    def vel_callback(self, data):
        self.vel_x = np.float32(data.linear.x)
        self.vel_y = np.float32(data.linear.y)
        self.vel_z = np.float32(data.linear.z)
        self.vel = np.array([self.vel_x, self.vel_y, self.vel_z])

    def gripper_callback(self, data):
        self.gripper_width = np.float32(data.position[0])

    def process_image(self, data, depth=False):
        """Helper function to process RGB and depth images."""
        try:
            encoding = "16UC1" if depth else "rgb8"
            image = self.bridge.imgmsg_to_cv2(data, encoding)
        except CvBridgeError as e:
            print(e)
            return None

        crop_resolution = (min(image.shape[:2]), min(image.shape[:2]))
        if image.shape[:2] != crop_resolution:
            center = image.shape
            x = center[1] / 2 - crop_resolution[1] / 2
            y = center[0] / 2 - crop_resolution[0] / 2
            image = image[int(y):int(y + crop_resolution[0]), int(x):int(x + crop_resolution[1])]

        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(
                image,
                dsize=(self.width, self.height),
                interpolation=cv2.INTER_CUBIC,
            )
        return image

    def wrist1_callback(self, data):
        self.wrist1 = self.process_image(data)

    def wrist1_depth_callback(self, data):
        self.wrist1_depth = self.process_image(data, depth=True)

    def wrist2_callback(self, data):
        self.wrist2 = self.process_image(data)

    def wrist2_depth_callback(self, data):
        self.wrist2_depth = self.process_image(data, depth=True)

    def front_callback(self, data):
        self.front = self.process_image(data)

    def front_depth_callback(self, data):
        self.front_depth = self.process_image(data, depth=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Wait for fresh state
        while not self.are_attributes_initialized():
            rclpy.spin_once(self.node, timeout_sec=0.01)

        
        # Reset robot to initial pose
        pose = Pose()
        pose.position.x = float(self.initial_position[0])
        pose.position.y = float(self.initial_position[1])
        pose.position.z = float(self.initial_position[2])
        pose.orientation.x = float(self.initial_orientation[0])
        pose.orientation.y = float(self.initial_orientation[1])
        pose.orientation.z = float(self.initial_orientation[2])
        pose.orientation.w = float(self.initial_orientation[3])

        self.goal_pose_pub.publish(pose)
        self.gripper_pub.publish(Float32(data=-1.0))

        time.sleep(5)
        
        # Determine if current pose is close to initial pose
        current_pos = np.array([self.x, self.y, self.z])
        current_orientation = Rotation.from_matrix(self.rot_mat).as_quat()
        pos_diff = np.linalg.norm(current_pos - self.initial_position)
        rot_diff = np.abs(np.dot(current_orientation, self.initial_orientation))
        # Loop until robot is near initial pose
        while (pos_diff > 0.01) or (rot_diff < 0.99):
            current_pos = np.array([self.x, self.y, self.z])
            current_orientation = Rotation.from_matrix(self.rot_mat).as_quat()
            pos_diff = np.linalg.norm(current_pos - self.initial_position)
            rot_diff = np.abs(np.dot(current_orientation, self.initial_orientation))
            self.goal_pose_pub.publish(pose)
            self.gripper_pub.publish(Float32(data=-1.0))
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.1)

        time.sleep(1)
        
        self.prev_action = np.zeros(self.action_space.shape)
        self.last_step_time = time.time()
        self.prev_gripper_state = 0 # 0 for open, 1 for closed
        self.gripper_state = 0

        return self._get_obs(), {}

    def step(self, action):
        # Check if environment is initialized
        if not self.are_attributes_initialized():
            raise RuntimeError("Environment not properly initialized")
        
        # Seems to perform better than while loop comparing current state to previous state     
        for i in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # clip, parse and scale actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.ee_dof == 3:
            z, y, x, grasp = action
            roll, pitch, yaw = 0, 0, 0
        elif self.ee_dof == 4:
            z, y, x, yaw, grasp = action
            roll, pitch = 0, 0
        elif self.ee_dof == 6:
            z, y, x, roll, pitch, yaw, grasp = action
        dpos = np.array([x, y, z]) * self.pos_scale
        drot = np.array([roll, pitch, yaw]) * self.rot_scale

        current_rotation = Rotation.from_matrix(self.rot_mat)
        # dpos_world = [dpos[2], dpos[1], dpos]
        dpos_world = current_rotation.apply(dpos)        
        # Create pose message
        pose = Pose()
        # Apply position change
        pose.position.x = self.x + float(dpos_world[0])
        pose.position.y = self.y + float(dpos_world[1])
        pose.position.z = self.z + float(dpos_world[2])
        # Clip position to bounds
        pose.position.x = np.clip(pose.position.x, self._CARTESIAN_BOUNDS[0, 0], self._CARTESIAN_BOUNDS[1, 0])
        pose.position.y = np.clip(pose.position.y, self._CARTESIAN_BOUNDS[0, 1], self._CARTESIAN_BOUNDS[1, 1])
        pose.position.z = np.clip(pose.position.z, self._CARTESIAN_BOUNDS[0, 2], self._CARTESIAN_BOUNDS[1, 2])

        # Apply rotation change
        current_rotation = Rotation.from_matrix(self.rot_mat)
        # Sim quat is in wxyz, need to match this
        action_rotation = Rotation.from_euler('xyz', drot)
        new_rotation = action_rotation * current_rotation
        new_relative_rotation = self.initial_rotation.inv() * new_rotation
        relative_euler = new_relative_rotation.as_euler('xyz')
        clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
        clipped_rotation = Rotation.from_euler('xyz', clipped_euler)
        final_rotation = self.initial_rotation * clipped_rotation
        final_quat = final_rotation.as_quat()
        pose.orientation.x = float(final_quat[0])
        pose.orientation.y = float(final_quat[1])
        pose.orientation.z = float(final_quat[2])
        pose.orientation.w = float(final_quat[3])

        # Handle grasping
        if time.time() - self.prev_grasp_time < 0.5:
            self.gripper_blocked = True
        else:
            if grasp >= 0.5:
                if self.prev_grasp >=0.5:
                    pass
                else:
                    width = 0.0
                    self.gripper_pub.publish(Float32(data=float(width)))
                    self.prev_grasp_time = time.time()
                    self.prev_grasp = grasp
                    self.gripper_vec = self.gripper_dict["closing"]
            elif grasp <= -0.5:
                width = np.clip(2*self.gripper_width + 0.005, 0.0, 0.076)
                self.gripper_pub.publish(Float32(data=float(width)))
                self.prev_grasp_time = time.time()
                self.prev_grasp = grasp
                self.gripper_vec = self.gripper_dict["opening"]
            else:
                self.gripper_blocked = False
                self.prev_grasp = grasp
                self.gripper_vec = self.gripper_dict["stopped"]

        self.goal_pose_pub.publish(pose)
                
        # Calculate reward
        reward, info = self._compute_reward(action)
        # Update previous action
        self.prev_action = action
        
        # Check for success/termination
        terminated = False
        truncated = False
        if time.time() - self.last_step_time < self.control_dt:
            time.sleep(self.control_dt - (time.time() - self.last_step_time))
        self.last_step_time = time.time()
        rclpy.spin_once(self.node, timeout_sec=0.01)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Wait for fresh state
        while not self.are_attributes_initialized():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        obs = {"state": {}}
        
        # State observations
        obs["state"]["panda/tcp_pos"] = np.array([self.x, self.y, self.z], dtype=np.float32)
        quat = Rotation.from_matrix(self.rot_mat).as_quat().astype(np.float32)
  
        obs["state"]["panda/tcp_orientation"] = quat
        obs["state"]["panda/tcp_vel"] = self.vel
        obs["state"]["panda/gripper_pos"] = (25*2*np.array([self.gripper_width], dtype=np.float32)-1)
        obs["state"]["panda/gripper_vec"] = np.concatenate([self.gripper_vec, [self.grasp], [int(self.gripper_blocked)]]).astype(np.float32)
        
        if self.image_obs:
            obs["images"] = {camera: getattr(self, camera) for camera in self.cameras}
            if self.depth:
                obs["images"].update({f"{camera}_depth": getattr(self, f"{camera}_depth") for camera in self.cameras})

            
        return obs

    def _compute_reward(self, action):
        # Basic reward based on smoothness
        action_diff = np.linalg.norm(action[:-1] - self.prev_action[:-1]) / np.sqrt(len(action)-1)
        smooth_reward = 1 - np.tanh(5 * action_diff)
        
        reward = smooth_reward
        
        info = {
            'smooth_reward': smooth_reward
        }
        
        return reward, info

    def close(self):
        rclpy.shutdown()

    def render(self):
            if self.image_obs:
                if self.are_attributes_initialized():
                    return [getattr(self, camera) for camera in self.cameras]

            else:
                return [np.zeros((self.height, self.width, 3), dtype=np.uint8), np.zeros((self.height, self.width, 3), dtype=np.uint8)]

    def __del__(self):
        self.close()