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
ros2 launch realsense2_camera rs_launch.py serial_no:=_048122070681 camera_namespace:=/ camera_name:=front align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30

ros2 launch realsense2_camera rs_launch.py serial_no:=_801212071197 camera_namespace:=/ camera_name:=wrist1 align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30

ros2 launch realsense2_camera rs_launch.py serial_no:=_752112070781 camera_namespace:=/ camera_name:=wrist2 align_depth.enable:=true rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30

### If not using depth

ros2 launch realsense2_camera rs_launch.py   serial_no:=_752112070781   camera_namespace:=/   camera_name:=wrist2   enable_depth:=false   enable_infra1:=false   enable_infra2:=false   align_depth.enable:=false   rgb_camera.color_profile:=640x480x60

ros2 launch realsense2_camera rs_launch.py serial_no:=_801212071197 camera_namespace:=/ camera_name:=wrist1 enable_depth:=false   enable_infra1:=false   enable_infra2:=false   align_depth.enable:=false   rgb_camera.color_profile:=640x480x60



### Simulation Environment
- install gym_INB0104 https://github.com/emlynw/gym_INB0104


