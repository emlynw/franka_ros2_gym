### Franka Gymnasium Environment Using ROS 2

A Gymnasium Environment for Using the Franka Panda Robot with Ros 2

### Installation:

Tested on FCI 4.2.1 using Libfranka 0.9.2 with ubuntu 22.04 (ROS Humble)
Install https://github.com/emlynw/multipanda_ros2 on the Realtime Kernel PC connected to the franka. 

clone the repository
cd franka_ros2_gym
pip install -e .

### Example Usage
On the Franka PC: 
Run realsense cameras: 
ros2 launch realsense2_camera rs_launch.py serial_no:=_048122070681 camera_namespace:=camera1 camera_name:=camera1 align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30

ros2 launch realsense2_camera rs_launch.py serial_no:=_801212071197 camera_namespace:=camera2 camera_name:=camera2 align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30

ros2 launch realsense2_camera rs_launch.py serial_no:=_752112070781 camera_namespace:=camera2 camera_name:=camera2 align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30


### editing and adding to the simulation:
- The gymnasium simulation environments are found in /gym_INB0104/gym_INB0104/envs


