#include <Arduino.h>
#include <Stepper.h>

// Number of steps per output rotation
const int stepsPerRevolution = 200;

// Create Instance of Stepper library
Stepper myStepper(stepsPerRevolution, 2, 13, 14, 15);

void setup()
{
  // set the speed at 20 rpm:
  myStepper.setSpeed(40);
  // initialize the serial port:
  Serial.begin(115200);
}

void loop()
{
  // step one revolution in one direction:
  Serial.println("clockwise");
  myStepper.step(stepsPerRevolution);
  delay(500);

  // step one revolution in the other direction:
  Serial.println("counterclockwise");
  myStepper.step(-stepsPerRevolution);
  delay(500);
}