import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict

import rclpy
from rclpy.node import Node
from rclpy import qos
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from franka_msgs.msg import FrankaState, FrankaModel
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from controller_manager_msgs.srv import SwitchController

import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation, Slerp
import time
from scipy.spatial.transform import Rotation as R

class ReachIKAbsRealStrawbEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        image_obs=True,
        ee_dof=6,  # 3 for position only, 4 for position+yaw
        width=640,
        height=480,
        pos_scale=0.01,
        rot_scale=0.2,
        control_dt=0.05,
        cameras=["wrist1", "wrist2"],
        depth = False,
        randomize_domain=False,
        gripper_pause=False,
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
        self.randomize_domain = randomize_domain
        self.gripper_pause = gripper_pause
        self.gripper_sleep = 0.6
        # Control parameters
        # self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
        # self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/10, -np.pi/10],[np.pi/3, np.pi/10, np.pi/10]], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.0, -0.4, 0.0], [0.9, 0.4, 0.95]], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([[-np.pi, -np.pi, -np.pi],[np.pi, np.pi, np.pi]], dtype=np.float32)
        self.ee_noise_low = [-0.04, -0.05, 0.0]
        self.ee_noise_high = [0.04, 0.05, 0.1]

        # Initial poses
        self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
        self.initial_orientation = [0.725, 0.0, 0.688, 0.0]
        # self.initial_position = np.array([0.3, 0.0, 0.5], dtype=np.float32)
        # self.initial_orientation = [1, 0.0, 0.0, 0.0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)
        
        # Action and observation spaces
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32
        )
        
        state_space = Dict({
            "tcp_pose": Box(
                -np.inf, 
                np.inf, 
                shape=(7,), 
                dtype=np.float32
            ),
            "tcp_vel": Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
            # "gripper_vec": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
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
        
        
        # ROS setup
        self.required_attributes = ["pos", "gripper_width"] + [
            camera for camera in self.cameras
        ] + [f"{camera}_depth" for camera in self.cameras if self.depth]
        for attr in self.required_attributes:
            setattr(self, attr, None)

        # Initialize attributes to None
        for attr in self.required_attributes:
            setattr(self, attr, None)
        
        self.prev_grasp_time = time.time()
        self.prev_grasp = -1.0
        self.gripper_dict = {
            "stopped": np.array([1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 1], dtype=np.float32),
        }
        # self.gripper_vec = self.gripper_dict["stopped"]
        self.grasp = -1.0
        self.gripper_blocked = False
        self.gripper_open_width = 0.08
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
        tmatrix = np.array(list(data.o_t_ee)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(data.dq)).reshape((7,))
        self.q = np.array(list(data.q)).reshape((7,))
        self.force = np.array(list(data.k_f_ext_hat_k)[:3])
        self.torque = np.array(list(data.k_f_ext_hat_k)[3:])   
        
    def vel_callback(self, data):
        self.vel = np.array([np.float32(data.linear.x), np.float32(data.linear.y), np.float32(data.linear.z), 
                             np.float32(data.angular.x), np.float32(data.angular.y), np.float32(data.angular.z)])

    def gripper_callback(self, data):
        self.gripper_width = np.float32(np.sum(data.position) / self.gripper_open_width)

    def process_image(self, data, depth=False):
        """Helper function to process RGB and depth images."""
        try:
            encoding = "16UC1" if depth else "rgb8"
            image = self.bridge.imgmsg_to_cv2(data, encoding)
        except CvBridgeError as e:
            print(e)
            return None
        h,w = image.shape[:2]
        if depth:
            crop_resolution = (h, w)
        else:
            crop_resolution = (h, w)
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
        
        # Step 1: Deactivate the current controller
        switch_controller_client = self.node.create_client(SwitchController, '/controller_manager/switch_controller')
        if not switch_controller_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Service /controller_manager/switch_controller not available")
        
        # Determine the active controller (initialize if not set)
        if not hasattr(self, "active_controller"):
            self.active_controller = "cartesian_impedance_controller"
        
        current_controller = self.active_controller

        # Request to deactivate current controller and activate the move_to_start_controller
        switch_request = SwitchController.Request()
        print(f"Resetting Joint Positions")
        switch_request.deactivate_controllers = [current_controller]
        time.sleep(1)
        switch_request.activate_controllers = ["move_to_start_controller"]
        switch_request.strictness = SwitchController.Request.STRICT
        
        future = switch_controller_client.call_async(switch_request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is None or not future.result().ok:
            raise RuntimeError("Failed to switch controllers to move_to_start_controller")
        else:
            # Update the active controller
            self.active_controller = "move_to_start_controller"
        
        # Step 2: Wait for the move_to_start_controller to complete
        time.sleep(2)  # Adjust based on your controller's behavior

        # Open gripper
        # for i in range(20):
        #     self.gripper_pub.publish(Float32(data=0.0))
        for i in range(20):
            self.gripper_pub.publish(Float32(data=self.gripper_open_width))
        self.prev_grasp_time = time.time()

        
        # Step 3: Switch back to the previous controller
        switch_request = SwitchController.Request()
        switch_request.deactivate_controllers = ["move_to_start_controller"]
        time.sleep(1)
        switch_request.activate_controllers = [current_controller]
        switch_request.strictness = SwitchController.Request.STRICT
        
        future = switch_controller_client.call_async(switch_request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is None or not future.result().ok:
            raise RuntimeError(f"Failed to switch back to {current_controller}")
        else:
            # Restore the previous controller as active
            self.active_controller = current_controller
        
        # Reset robot to initial pose
        start_pos = self.pos[0:3]
        start_quat = self.pos[3:]
        if self.randomize_domain:
            target_pos = self.initial_position + np.random.uniform(low=self.ee_noise_low, high=self.ee_noise_high, size=3)
            target_pos = np.clip(target_pos, self._CARTESIAN_BOUNDS[0], self._CARTESIAN_BOUNDS[1])
        else:
            target_pos = self.initial_position
        target_quat = self.initial_orientation

        start_rot = Rotation.from_quat(start_quat)
        target_rot = Rotation.from_quat(target_quat)

        # Create Slerp interpolator
        slerp = Slerp([0, 1], Rotation.concatenate([start_rot, target_rot]))

        # Interpolation parameters
        reset_duration = 3.0  # Total time for smooth interpolation
        num_steps = int(reset_duration / self.control_dt)

        # Smooth interpolation trajectory
        for step in range(num_steps):
            alpha = (step + 1) / num_steps
            
            # Linear position interpolation
            interp_pos = (1 - alpha) * start_pos + alpha * target_pos
            
            # Spherical rotation interpolation
            interp_rot = slerp(alpha)
            interp_quat = interp_rot.as_quat()

            # Create and publish intermediate pose
            pose = Pose()
            pose.position.x = float(interp_pos[0])
            pose.position.y = float(interp_pos[1])
            pose.position.z = float(interp_pos[2])
            pose.orientation.x = float(interp_quat[0])
            pose.orientation.y = float(interp_quat[1])
            pose.orientation.z = float(interp_quat[2])
            pose.orientation.w = float(interp_quat[3])
            
            self.goal_pose_pub.publish(pose)
            
            # Maintain control rate and process updates
            time.sleep(self.control_dt)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        # Final verification loop
        start_time = time.time()
        while time.time() - start_time < 5.0:  # Max 5s verification
            current_pos = self.pos[0:3]
            current_quat = self.pos[3:]
            
            pos_diff = np.linalg.norm(current_pos - target_pos)
            rot_diff = np.abs(np.dot(current_quat, target_quat))
            
            if pos_diff < 0.01 and rot_diff > 0.99:
                break
                
            # Continue publishing final target pose
            self.goal_pose_pub.publish(pose)
            time.sleep(0.1)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        time.sleep(1)
        
        self.prev_action = np.zeros(self.action_space.shape)
        self.last_step_time = time.time()
        self.prev_gripper_state = 0 # 0 for open, 1 for closed
        self.gripper_state = 0

        self.origin_pos = self.pos[0:3].copy()                 # (x0,y0,z0)
        self.origin_rot = Rotation.from_quat(self.pos[3:])  

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

        target_rel_pos = np.array([x, y, z]) * self.pos_scale
        target_world_pos = self.origin_pos + target_rel_pos

        target_rel_rot   = Rotation.from_euler('xyz', np.array([roll, pitch, yaw]) * self.rot_scale)
        target_world_rot = self.origin_rot * target_rel_rot  

        target_world_pos = np.clip(target_world_pos, self._CARTESIAN_BOUNDS[0], self._CARTESIAN_BOUNDS[1])

        # Create pose message
        pose = Pose()
        # Apply position change
        pose.position.x = target_world_pos[0]
        pose.position.y = target_world_pos[1]
        pose.position.z = target_world_pos[2]

        # Apply rotation change
        pose.orientation.x, pose.orientation.y, \
        pose.orientation.z, pose.orientation.w = target_world_rot.as_quat()

        self.goal_pose_pub.publish(pose)

        # Handle grasping

        if (grasp >= 0.5) and (self.gripper_width > 0.85) and (time.time() - self.prev_grasp_time > self.gripper_sleep):  # close gripper
            width = 0.0
            self.gripper_pub.publish(Float32(data=float(width)))
            self.prev_grasp_time = time.time()
            if self.gripper_pause:
                time.sleep(self.gripper_sleep)
        elif (grasp <= -0.5) and (self.gripper_width < 0.85) and (time.time() - self.prev_grasp_time > self.gripper_sleep):  # open gripper
            self.gripper_pub.publish(Float32(data=float(self.gripper_open_width)))
            self.prev_grasp_time = time.time()
            if self.gripper_pause:
                time.sleep(self.gripper_sleep)
        
        # if time.time() - self.prev_grasp_time < 0.5:
        #     self.gripper_blocked = True
        # else:
        #     if grasp >= 0.5:
        #         if self.prev_grasp >=0.5:
        #             pass
        #         else:
        #             width = 0.0
        #             self.gripper_pub.publish(Float32(data=float(width)))
        #             self.prev_grasp_time = time.time()
        #             self.prev_grasp = grasp
        #             self.gripper_vec = self.gripper_dict["closing"]
        #     elif grasp <= -0.5:
        #         # width = np.clip(2*self.gripper_width + 0.005, 0.0, 0.076)
        #         self.gripper_pub.publish(Float32(data=float(self.gripper_open_width)))
        #         self.prev_grasp_time = time.time()
        #         self.prev_grasp = grasp
        #         self.gripper_vec = self.gripper_dict["opening"]
        #     else:
        #         self.gripper_blocked = False
        #         self.prev_grasp = grasp
        #         self.gripper_vec = self.gripper_dict["stopped"]
                
        # Calculate reward
        reward, info = self._compute_reward(action)
        # Update previous action
        self.prev_action = action
        
        # Check for success/termination
        terminated = False
        truncated = False
        control_step_time = time.time() - self.last_step_time
        if control_step_time < self.control_dt:
            time.sleep(self.control_dt - (control_step_time))
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
        obs["state"]["tcp_pose"] = self.pos
        obs["state"]["tcp_vel"] = self.vel
        obs["state"]["gripper_pos"] = np.array([self.gripper_width])
        # obs["state"]["gripper_vec"] = np.concatenate([self.gripper_vec, [int(self.gripper_blocked)]]).astype(np.float32)

        if self.image_obs:
            obs["images"] = {camera: getattr(self, camera) for camera in self.cameras}
            if self.depth:
                obs["images"].update({f"{camera}_depth": getattr(self, f"{camera}_depth") for camera in self.cameras})

            
        return obs

    def _compute_reward(self, action):
        # Basic reward based on smoothness
        action_diff = np.linalg.norm(action[:-1] - self.prev_action[:-1]) / np.sqrt(len(action)-1)
        smooth_reward = 1 - np.tanh(5 * action_diff)
        
        info = {
            'smooth_reward': smooth_reward
        }

        reward = 0
        
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