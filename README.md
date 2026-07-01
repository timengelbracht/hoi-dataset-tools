git clone git@github.com:timengelbracht/hoi-dataset-tools.git
cd hoi-dataset-tools


# recording docker build
mkdir source/recordings/
cd source/recordings/
docker compose build

# record data in rosbag
docker compose up -d
docker exec -it spot_aria_gripper_recorder /bin/bash
rosparam set use_sim_time false
roslaunch witmotion_ros wt901.launch 

new terminal
docker exec -it spot_aria_gripper_recorder /bin/bash
rosbag record -o /bags/imu_data /imu


# get imu noise level
record a rosbag for at least 3 hours etc use Allan Variance ROS https://github.com/ori-drs/allan_variance_ros
needed for kalibr (camera/imu calibration)
rosrun allan_variance_ros allan_variance /bags/cooked_rosbag.bag /hoi-dataset-tools/config/imu_noise/witmotion_imu.yaml

rosrun allan_variance_ros allan_variance /bags/cooked_rosbag.bag /hoi-dataset-tools/config/imu_noise/ witmotion_imu.yaml
rosrun allan_variance_ros allan_variance /bags/cooked_rosbag.bag /hoi-dataset-tools/config/imu_noise/witmotion_imu.yaml


create imu yaml
witmotion.yaml
rosbag record --duration=60s -o /bags/test /imu


note: imu polling rate set via software

# IMU noise stuff

# Lenai Gripper setup

# motor
dynamixel_workbench_controllers/config/basic.yaml ID=1
min/max position [1615,2536]
src/dynamixel-workbench/dynamixel_workbench_controllers/config/basic.yaml
pan:
  ID: 1
  Return_Delay_Time: 0
  Operating_Mode: 16
  Min_Position_Limit: 1630
  Max_Position_Limit: 2500

command line setting
rosservice call /dynamixel_workbench/dynamixel_command "{command: '', id: 1, addr_name: 'Goal_Position', value: 2048}"

# publish motor topics
roslaunch dynamixel_workbench_controllers dynamixel_controllers.launch
roslaunch dynamixel_sdk_examples read_write.launch dxl_port:=/dev/ttyUSB0 dxl_baud:=57600 dxl_id:=1

# IMU camera calib
todo kalibr setup


# trigger setup
#calibration
source/arduino/calibration
Place a known weight (e.g., 1 kg) on the load cell and record the reading.
Compute the scale factor:
If 1kg gives 10,000 raw reading, then scale factor = 10000/1kg = 10000.
Use this value in the next code.

# run ros/arduino recording
flash force_reading_ros onto arduino -> publishes /gripper_force_trigger
rosrun rosserial_python serial_node.py /dev/ttyUSB1 _baud:=57600

# run throslaunch gripper_force_controller gripper_control.launche motor controller (arduino needs to be running)
roslaunch gripper_force_controller gripper_control.launch


# open vins odometry
cd data_processing/docker/odometry
docker compose build
docker compose up -d
docker exec -it open_vins /bin/bash

roslaunch ov_msckf subscribe.launch config:=zedm bag:=/bags/zedm/cam_left_right_imu_calib.bag


# open zed ros dev
hoi-dataset-tools/zed_open_capture_ros

docker run -it --rm \
  --name zed_open_capture_dev \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v /home/Documents/hoi-dataset-tools/zed_open_capture_ros:/catkin_ws/src \
  zed_open_capture_dev


# gelsight digit

docker run -it --rm \
  --privileged \
  --device /dev/bus/usb \
  -v /dev/bus/usb:/dev/bus/usb \
  --group-add plugdev \
  your_digit_ros_image


apply udev rules on host

check out ros_noetic/dev_container/.devcontainer/devcontainer.json for run args, permissions, udev etc

roslaunch gelsight_digit_ros gelsight_digit_node.launch device_id:=D21237 topic_name:=/digit/left/image_raw



cvg@cvg-System-Product-Name:~/Documents/hoi-dataset-tools/data_processing/docker/odometry$ ls /dev/serial/by-id/
usb-FTDI_FT232R_USB_UART_A10K4UM5-if00-port0
usb-FTDI_USB__-__Serial_Converter_FT89FCUV-if00-port0

