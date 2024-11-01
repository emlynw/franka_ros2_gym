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
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation
import time
import threading

class ReachIKDeltaRealEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        image_obs=True,
        ee_dof=4,  # 3 for position only, 4 for position+yaw
        width=224,
        height=224,
        pos_scale=0.1,
        rot_scale=0.05,
        **kwargs
    ):
        super().__init__()

        # Environment parameters
        self.image_obs = image_obs
        self.ee_dof = ee_dof
        self.width = width
        self.height = height
        
        # Action and observation spaces
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32
        )
        
        state_space = Dict({
            "panda/tcp_pos": Box(
                np.array([0.28, -0.35, 0.01]), 
                np.array([0.75, 0.35, 0.55]), 
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
            self.observation_space["images"] = Dict({
                "front": Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
            })
        
        # Control parameters
        self._CARTESIAN_BOUNDS = np.array([
            [0.28, -0.35, 0.01],
            [0.75, 0.35, 0.55]
        ], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([
            [-np.pi/4, -np.pi/4, -np.pi/2],
            [np.pi/4, np.pi/4, np.pi/2]
        ], dtype=np.float32)
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        
        # Initial poses
        self.initial_position = np.array([0.3, 0.0, 0.5], dtype=np.float32)
        self.initial_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)
        
        # State variables
        self.state_initialized = False
        self.gripper_initialized = False
        self.image_initialized = False
        self.resetting = False
        self.image = None
        
        # Gripper state tracking
        self.prev_grasp_time = 0.0
        self.prev_grasp = -1.0
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }
        self.gripper_vec = self.gripper_dict["open"]
        self.ros_setup()

    def ros_setup(self):
        # Initialize ROS node
        rclpy.init()
        self.node = rclpy.create_node('reach_ik_delta_real')

         # Add callback group and executor for better callback handling
        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
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
        
        self.state_sub = self.node.create_subscription(
            FrankaState, 
            '/franka_robot_state_broadcaster/robot_state',
            self.state_callback,
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
        self.image_sub = self.node.create_subscription(
            Image,
            '/color/image_raw',
            self.image_callback,
            qos_profile=custom_qos_profile,
            callback_group=self.callback_group
        )

        # Start executor in separate thread
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self._run_executor)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        # Add state lock for thread safety
        self.state_lock = threading.Lock()
        self.last_state = None
        self._latest_state = None

        self.bridge = CvBridge()

    def _run_executor(self):
        try:
            while rclpy.ok():
                self.executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            print(f"Executor exception: {e}")

    def state_callback(self, data):
        with self.state_lock:
            self.rot_mat = np.array([
                [data.o_t_ee[0], data.o_t_ee[4], data.o_t_ee[8]],
                [data.o_t_ee[1], data.o_t_ee[5], data.o_t_ee[9]],
                [data.o_t_ee[2], data.o_t_ee[6], data.o_t_ee[10]]
            ])
            self.x = np.float32(data.o_t_ee[12])
            self.y = np.float32(data.o_t_ee[13])
            self.z = np.float32(data.o_t_ee[14])
            self._latest_state = data
            self.state_initialized = True
            print(f"state: {self.x}, {self.y}, {self.z}")

    def gripper_callback(self, data):
        self.gripper_width = np.float32(data.position[0]*2)
        self.gripper_initialized = True

    def image_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(
                image,
                dsize=(self.width, self.height),
                interpolation=cv2.INTER_CUBIC,
            )
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        self.image = image
        self.image_initialized = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        print("Resetting robot...")
        self.resetting = True
        
        # Reset robot to initial pose
        pose = Pose()
        pose.position.x = float(self.initial_position[0])
        pose.position.y = float(self.initial_position[1])
        pose.position.z = float(self.initial_position[2])
        pose.orientation.x = float(self.initial_orientation[0])
        pose.orientation.y = float(self.initial_orientation[1])
        pose.orientation.z = float(self.initial_orientation[2])
        pose.orientation.w = float(self.initial_orientation[3])
        
        # Send reset commands
        for _ in range(5):
            self.goal_pose_pub.publish(pose)
            self.gripper_pub.publish(Float32(data=-1.0))
            # rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.1)
        
        # Wait for robot to reach position
        time.sleep(4.0)
        
        self.resetting = False
        self.state_initialized = False
        self.prev_action = np.zeros(self.action_space.shape)
        
        # Wait for fresh state
        while not (self.state_initialized and self.gripper_initialized and self.image_initialized):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return self._get_obs(), {}

    def step(self, action):
        if not (self.state_initialized and self.gripper_initialized and self.image_initialized):
            raise RuntimeError("Environment not properly initialized")
        # for i in range(5):
        #     rclpy.spin_once(self.node, timeout_sec=0.01)

        # Wait for next state update with timeout
        start_time = time.time()

        while time.time() - start_time < 1.0:  # 1 second timeout
            with self.state_lock:
                if self._latest_state != self.last_state:
                    print("State updated")
                    self.last_state = self._latest_state
                    break
            time.sleep(0.01)
        
        # Get new observation
        obs = self._get_obs()
            
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Parse and scale actions
        if self.ee_dof == 3:
            x, y, z, grasp = action
            roll, pitch, yaw = 0, 0, 0
        elif self.ee_dof == 4:
            x, y, z, yaw, grasp = action
            roll, pitch = 0, 0
        
        dpos = np.array([x, y, z]) * self.pos_scale
        print(f"dpos: {dpos}")
        drot = np.array([roll, pitch, yaw]) * self.rot_scale
        
        # Create pose message
        pose = Pose()
        
        print(f"old pos: {self.x}, {self.y}, {self.z}")
        # Apply position change
        pose.position.x = self.x + dpos[0]
        pose.position.y = self.y + dpos[1]
        pose.position.z = self.z + dpos[2]
        print(f"new pos: {pose.position.x}, {pose.position.y}, {pose.position.z}")
        
        # Clip position to bounds
        pose.position.x = np.clip(pose.position.x, self._CARTESIAN_BOUNDS[0, 0], self._CARTESIAN_BOUNDS[1, 0])
        pose.position.y = np.clip(pose.position.y, self._CARTESIAN_BOUNDS[0, 1], self._CARTESIAN_BOUNDS[1, 1])
        pose.position.z = np.clip(pose.position.z, self._CARTESIAN_BOUNDS[0, 2], self._CARTESIAN_BOUNDS[1, 2])

        print(f"clipped pos: {pose.position.x}, {pose.position.y}, {pose.position.z}")
        
        # Apply rotation change
        current_rotation = Rotation.from_matrix(self.rot_mat)
        action_rotation = Rotation.from_euler('xyz', drot)
        new_rotation = action_rotation * current_rotation
        new_relative_rotation = self.initial_rotation.inv() * new_rotation
        relative_euler = new_relative_rotation.as_euler('xyz')
        clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
        clipped_rotation = Rotation.from_euler('xyz', clipped_euler)
        final_rotation = self.initial_rotation * clipped_rotation
        final_quat = final_rotation.as_quat()
        
        pose.orientation.x = final_quat[0]
        pose.orientation.y = final_quat[1]
        pose.orientation.z = final_quat[2]
        pose.orientation.w = final_quat[3]
        
        # Handle gripper control
        grasp_cmd = float(grasp > 0)
        if self.node.get_clock().now().nanoseconds/1e9 - self.prev_grasp_time < 0.5:
            grasp_cmd = self.prev_grasp
        else:
            self.prev_grasp_time = self.node.get_clock().now().nanoseconds/1e9
            self.prev_grasp = grasp_cmd
            
            if grasp_cmd > 0:
                self.gripper_vec = self.gripper_dict["closing"]
            else:
                self.gripper_vec = self.gripper_dict["opening"]
        
        # Publish commands
        print(f"a: {action}")
        print(f"pose: {pose.position.x}, {pose.position.y}, {pose.position.z}, {pose.orientation.x}, {pose.orientation.y}, {pose.orientation.z}, {pose.orientation.w}")
        for i in range(5):
            self.goal_pose_pub.publish(pose)
        self.gripper_pub.publish(Float32(data=float(grasp_cmd)))
        
        
        # Calculate reward
        reward, info = self._compute_reward(action)
        
        # Update previous action
        self.prev_action = action
        
        # Check for success/termination
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Wait for fresh state
        while not (self.state_initialized and self.gripper_initialized and self.image_initialized):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        obs = {"state": {}}
        
        # State observations
        obs["state"]["panda/tcp_pos"] = np.array([self.x, self.y, self.z], dtype=np.float32)
        obs["state"]["panda/tcp_orientation"] = Rotation.from_matrix(self.rot_mat).as_quat().astype(np.float32)
        obs["state"]["panda/tcp_vel"] = np.zeros(3, dtype=np.float32)  # Velocity not available
        obs["state"]["panda/gripper_pos"] = np.array([self.gripper_width], dtype=np.float32)
        obs["state"]["panda/gripper_vec"] = self.gripper_vec
        
        if self.image_obs:
            obs["images"] = {
                "front": self.image
            }
            
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
        self.executor.shutdown()
        rclpy.shutdown()
        if self.executor_thread.is_alive():
            self.executor_thread.join(timeout=1.0)

    def render(self):
            if self.image_obs and self.image_initialized:
                return self.image
            else:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def __del__(self):
        self.close()