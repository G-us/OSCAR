/* Example sketch to control a stepper motor with L298N motor driver, Arduino UNO and AccelStepper.h library.
   Contiuous rotation.
   https://www.makerguides.com */
#include <Arduino.h>
// Include the AccelStepper library:
#include "AccelStepper.h"

// Define the AccelStepper interface type:
#define MotorInterfaceType 4

// Create a new instance of the AccelStepper class:
AccelStepper stepper = AccelStepper(MotorInterfaceType, 22, 23, 18, 19);

void setup()
{
  // Set the maximum speed in steps per second:
  stepper.setMaxSpeed(1000);
}

void loop()
{
  // Set the speed of the motor in steps per second:
  stepper.setSpeed(-500);
  // Step the motor with constant speed as set by setSpeed():
  stepper.runSpeed();
}