# Use ROS Noetic as the base
FROM ros:noetic

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (ROS + system)
RUN apt-get update && apt-get install -y \
    ros-noetic-rosbag \
    python3-pip \
    git \
    build-essential \
    cmake \
    nano \
    curl \
    mesa-utils \
    ros-noetic-plotjuggler-ros \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-dynamixel-workbench-controllers \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial \
    python3-catkin-tools \
    v4l-utils \
    udev \
    libusb-1.0-0-dev \ 
    libhidapi-libusb0 \ 
    libhidapi-dev \ 
    libopencv-dev \
    libopencv-viz-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", SYMLINK+="gripper"' > /etc/udev/rules.d/99-usb-serial.rules \
    && echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="imu"' >> /etc/udev/rules.d/99-usb-serial.rules
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Add Intel RealSense repository & install dependencies
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && \
    apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Clone ROS repositories
RUN git clone --recursive https://github.com/ElettraSciComp/witmotion_IMU_ros.git src/witmotion_ros
#RUN git clone --recursive https://github.com/ori-drs/allan_variance_ros.git src/allan_variance_ros
RUN git clone --recursive --branch noetic-devel https://github.com/ROBOTIS-GIT/dynamixel-workbench-msgs.git src/dynamixel-workbench-msgs
RUN git clone --recursive --branch noetic-devel https://github.com/ROBOTIS-GIT/dynamixel-workbench.git src/dynamixel-workbench
#RUN git clone --recursive https://github.com/stereolabs/zed-ros-wrapper.git src/zed-ros-wrapper
RUN git clone https://github.com/stereolabs/zed-open-capture.git src/zed-open-capture
RUN git clone https://github.com/jypjypjypjyp/zed-open-capture-ros.git src/zed-open-capture-ros

# Install ZED Open Capture
WORKDIR /catkin_ws/src/zed-open-capture/udev
RUN bash install_udev_rule.sh
WORKDIR /catkin_ws/src/zed-open-capture
RUN mkdir build && cd build && cmake .. && make -j$(nproc)
RUN cd build && sudo make install && sudo ldconfig
WORKDIR /catkin_ws

# Securely Clone Bitbucket Repos
RUN mkdir -p /root/.ssh
COPY id_ed25519 /root/.ssh/id_ed25519
COPY id_ed25519.pub /root/.ssh/id_ed25519.pub
RUN chmod 600 /root/.ssh/id_ed25519
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone git@bitbucket.org:leggedrobotics/lenai_gripper.git src/lenai_gripper
RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone git@bitbucket.org:leggedrobotics/realtime_urdf_filter.git src/realtime_urdf_filter
RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone git@bitbucket.org:leggedrobotics/tactile_tools.git src/tactile_tools
RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone git@github.com:timengelbracht/realsense-ros.git src/realsense-ros
RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone git@github.com:timengelbracht/gripper_force_controller.git src/gripper_force_controller

# Modify IMU configuration file
RUN sed -i 's/use_native_orientation: true/use_native_orientation: false/' /catkin_ws/src/witmotion_ros/config/wt901.yml
RUN sed -i 's/port: ttyUSB0/port: imu/' /catkin_ws/src/witmotion_ros/config/wt901.yml
RUN sed -i 's/port: ttyUSB0/port: gripper/' /catkin_ws/src/lenai_gripper/lenai_gripper/config/teleop_setup.yaml 
# Install ROS dependencies
RUN apt-get update && rosdep update
RUN rosdep install --from-paths src --ignore-src -r -y || true

# Install Conda (for Tactile Tools)
#RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash miniconda.sh -b -p /opt/conda && \
#    rm miniconda.sh
#ENV PATH="/opt/conda/bin:$PATH"

#RUN /opt/conda/bin/conda init bash

# Create & Activate Conda Environment for Tactile Tools
#RUN conda create -n tactile python=3.8 -y
#SHELL ["conda", "run", "-n", "tactile", "/bin/bash", "-c"]

# Install Python dependencies for Tactile Tools
#WORKDIR /catkin_ws/src/tactile_tools
#RUN /opt/conda/bin/conda run -n tactile pip install -r contour_following/requirements.txt
#RUN /opt/conda/bin/conda run -n tactile conda install libffi==3.3 -y

WORKDIR /catkin_ws/src/tactile_tools/contour_following
RUN pip install -r requirements.txt \
    && apt-get install -y libffi-dev \
    && pip install cffi \
    && pip install torch open3d 

# Build the workspace with catkin build
WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build"

# Set up the environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
#RUN echo "conda activate tactile" >> ~/.bashrc

# Remove SSH keys for security
#RUN rm -rf /root/.ssh

# Default command (Keeps container running for interaction)
CMD ["bash"]
