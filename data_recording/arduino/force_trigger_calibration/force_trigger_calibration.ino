#include "HX711.h"

#define DOUT 19  // HX711 Data
#define SCK 18   // HX711 Clock

HX711 scale;

void setup() {
    Serial.begin(115200);
    scale.begin(DOUT, SCK);

    Serial.println("Remove all weight. Taring in 3 seconds...");
    delay(3000);
    scale.tare();  // Set zero reference
    Serial.println("Tare complete.");
}

void loop() {
    Serial.print("Raw HX711 Reading: ");
    Serial.println(scale.get_units());
    
    delay(500);
}
