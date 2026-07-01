# Gripper Force Trigger – Setup & Calibration Guide

This README explains how to install the load‑cell “trigger” on the gripper, calibrate it, and stream force readings to ROS.



### Required Components
- **Single‑point load cell** (rated for expected force range)
- **HX711** amplifier board
- **Arduino** (tested with Arduino Uno)
- Trigger mount (3‑D printed or machined)
- Mounting hardware (M4 bolts, nuts, washers)
- **Reference weight** (e.g., 1 kg mass)

### Signal Flow
```
[Load Cell] ──> [HX711] ──> [Arduino] ──> [/gripper_force_trigger (ROS topic)]
```

---
## 2. Mechanical Installation
Mount the trigger so that **one end of the load cell is rigidly fixed and the other end “floats”** (i.e., carries the load without touching other structures).

---
## 3. Calibration 

1. **Flash** `calibration.ino` onto the Arduino.  
2. **Open the Serial Monitor** (make sure baudrate matches). Wait until the sketch completes automatic **taring**—raw output should hover near 0.  
3. **Apply a known weight** to the floating end (e.g., 1 kg). Record the raw value displayed.

### 3.1 Compute the Scale Factor
```
SCALE_FACTOR = RAW_VALUE / (G × WEIGHT)
```
- `RAW_VALUE` – serial reading with the weight applied  
- `WEIGHT` – mass in kilograms  
- `G` – 9.81 m s⁻² (standard gravity)

**Example**  
If a 1 kg weight yields `RAW_VALUE = 100 000` counts:
```
SCALE_FACTOR = 100000 / (1 × 9.81) ≈ 10193.7
```
4. Open `force_reading_ros.ino` and set:
```cpp
#define SCALE_FACTOR  10193.7   // counts per newton
```
5. **Flash** `force_reading_ros.ino` onto the Arduino.

> **Re‑calibrate** whenever you change the mount, load cell, or ambient temperature.

---
## 4. Streaming Data to ROS
`force_reading_ros.ino` publishes a single `float` (force in newtons) on:
```
/gripper_force_trigger
```

### 4.1 Stand‑alone Test
```bash
rosrun rosserial_python serial_node.py /dev/ttyUSB1 _baud:=57600
rostopic echo /gripper_force_trigger
```

### 4.2 Integrated Gripper Recording
The gripper’s main launch file automatically starts the serial node, so no extra steps are required.

---
## 5. Example Workflow
```bash
# --- Calibration ---
arduino --upload calibration/calibration.ino
# Observe RAW_VALUE ...

# --- Firmware with scale factor ---
# (Edit force_reading_ros.ino first)
arduino --upload firmware/force_reading_ros.ino

# --- Stream to ROS ---
roslaunch gripper_recording record.launch
```

