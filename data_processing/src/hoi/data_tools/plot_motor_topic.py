import rosbag
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import JointState

bag_path = "/bags/dlab_testing_2_2025-04-08-15-06-43.bag"  # ← Update this
joint_topic = "/joint_states"

timestamps = []
positions = []
currents = []

with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[joint_topic]):
        ts = msg.header.stamp.to_sec()
        pos = msg.position[0] if msg.position else 0.0
        effort = msg.effort[0] if msg.effort else 0.0
        if effort > 15:
            effort = 0.0
        timestamps.append(ts)
        positions.append(np.rad2deg(pos))  # Convert to degrees
        currents.append(effort * 1000)     # Convert A to mA

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(timestamps, positions, label="Position (deg)")
plt.ylabel("Angle (°)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(timestamps, currents, label="Current (mA)", color="orange")
plt.ylabel("Current (mA)")
plt.xlabel("Time (s)")
plt.legend()

plt.suptitle("Motor State Over Time")
plt.tight_layout()
plt.savefig("/bags/motor_state_plot.png", format="png")  # Save the plot as an image
plt.close()  # Close the figure to free up memory