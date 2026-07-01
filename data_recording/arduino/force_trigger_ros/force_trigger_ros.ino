#include <ros.h>
#include <std_msgs/Float32.h>
#include "HX711.h"

#define DOUT 19  
#define SCK 18   
#define SCALE_FACTOR 20938  // Replace with your calibration value
#define FORCE_CUTOFF 80  // Max force allowed (Newton)
#define FILTER_SIZE 3  // Number of samples for smoothing

HX711 scale;
ros::NodeHandle nh;  // ROS node handle
std_msgs::Float32 force_msg;  // ROS message (float)
ros::Publisher force_pub("gripper_force_trigger", &force_msg);  // Publisher to /gripper_force_trigger

float forceReadings[FILTER_SIZE];  // Filter buffer
int filterIndex = 0;

void setup() {
    Serial.begin(57600);
    nh.initNode();
    nh.advertise(force_pub);
    
    scale.begin(DOUT, SCK);
    
    Serial.println("Taring... Remove all weight.");
    delay(3000);
    scale.tare();
    Serial.println("Tare complete. Ready!");

    // Initialize filter
    for (int i = 0; i < FILTER_SIZE; i++) {
        forceReadings[i] = 0.0;
    }
}

void loop() {
    float rawForce = scale.get_units() / SCALE_FACTOR;  // Convert raw data to kg

    // Apply cutoff limit
    if (rawForce > FORCE_CUTOFF) {
        rawForce = FORCE_CUTOFF;
    } else if (rawForce < 0) {
        rawForce = 0;
    }

    // Add new value to filter buffer
    forceReadings[filterIndex] = rawForce;
    filterIndex = (filterIndex + 1) % FILTER_SIZE;

    // Compute moving average
    float filteredForce = 0;
    for (int i = 0; i < FILTER_SIZE; i++) {
        filteredForce += forceReadings[i];
    }
    filteredForce /= FILTER_SIZE;

    Serial.print("Filtered Force (kg): ");
    Serial.println(filteredForce, 3);

    // Publish force to ROS
    force_msg.data = filteredForce;
    force_pub.publish(&force_msg);

    nh.spinOnce();
    delay(20);
}
