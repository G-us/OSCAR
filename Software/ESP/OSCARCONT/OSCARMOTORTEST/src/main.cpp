/*
 * ESP32 Motor Controller with ESP-NOW
 * Controls 3 DC motors using L293N drivers
 * Receives commands wirelessly via ESP-NOW
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>

// ============== MOTOR CONFIGURATION ==============
// L293N Pin Definitions for 3 Motors
// Motor 1
#define MOTOR1_IN1 13
#define MOTOR1_IN2 12
#define MOTOR1_EN 14

// Motor 2
#define MOTOR2_IN1 27
#define MOTOR2_IN2 26
#define MOTOR2_EN 25

// Motor 3
#define MOTOR3_IN1 33
#define MOTOR3_IN2 32
#define MOTOR3_EN 15

// PWM Configuration
#define PWM_FREQ 1000    // 1kHz PWM frequency
#define PWM_RESOLUTION 8 // 8-bit resolution (0-255)
#define PWM_CHANNEL_1 0
#define PWM_CHANNEL_2 1
#define PWM_CHANNEL_3 2

// Safety Configuration
#define WATCHDOG_TIMEOUT 100 // Stop motors if no command for 100ms
#define ACCEL_LIMIT 5        // Max speed change per cycle (smoother control)

// Resilience Configuration
#define ESPNOW_INIT_RETRIES 5
#define ESPNOW_RETRY_DELAY_MS 250
#define PEER_ADD_RETRIES 3

// ============== ESP-NOW CONFIGURATION ==============
// MAC address of the bridge ESP32 (update with your actual MAC)
uint8_t bridgeMacAddress[] = {0x08, 0xB6, 0x1F, 0x29, 0xD6, 0x5C};

// Message structure - must match laptop/bridge
typedef struct motor_command
{
  int8_t motor1_speed; // -100 to +100 (percentage)
  int8_t motor2_speed; // -100 to +100 (percentage)
  int8_t motor3_speed; // -100 to +100 (percentage)
  uint8_t flags;       // Bit 0: emergency stop, Bit 1: enable
} motor_command;

motor_command currentCommand = {0, 0, 0, 0};
motor_command targetCommand = {0, 0, 0, 0};

// ============== STATE VARIABLES ==============
unsigned long lastCommandTime = 0;
int8_t actualSpeed[3] = {0, 0, 0}; // Current motor speeds after ramping
bool motorsEnabled = false;        // Fail-safe: start disabled until enable flag received
bool espNowReady = false;

// ============== FUNCTION PROTOTYPES ==============
void setupMotors();
void setupESPNow();
bool initESPNowWithRetry();
bool addPeerWithRetry(const uint8_t *peerMac);
void updateMotors();
void setMotorSpeed(uint8_t motorNum, int8_t speed);
void emergencyStop();
void onDataReceived(const uint8_t *mac, const uint8_t *data, int len);

void setup()
{
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 Motor Controller Starting ===");

  setupMotors();
  setupESPNow();

  Serial.println("=== System Ready ===");
}

void loop()
{
  // If ESP-NOW not ready, keep motors stopped and retry periodically
  if (!espNowReady)
  {
    emergencyStop();
    static unsigned long lastRetry = 0;
    if (millis() - lastRetry > 2000)
    {
      lastRetry = millis();
      setupESPNow();
    }
    delay(20);
    return;
  }

  // Watchdog: Stop motors if no command received recently
  if (millis() - lastCommandTime > WATCHDOG_TIMEOUT)
  {
    if (motorsEnabled)
    {
      Serial.println("WATCHDOG: No command received, stopping motors");
      emergencyStop();
      motorsEnabled = false;
    }
  }

  // If motors disabled, immediately clear target and actual speeds
  if (!motorsEnabled)
  {
    targetCommand.motor1_speed = 0;
    targetCommand.motor2_speed = 0;
    targetCommand.motor3_speed = 0;
    actualSpeed[0] = 0;
    actualSpeed[1] = 0;
    actualSpeed[2] = 0;
  }

  // Smooth speed ramping (prevents jerky movements)
  for (int i = 0; i < 3; i++)
  {
    int8_t targetSpeed = motorsEnabled ? targetCommand.motor1_speed : 0;
    if (i == 1)
      targetSpeed = motorsEnabled ? targetCommand.motor2_speed : 0;
    if (i == 2)
      targetSpeed = motorsEnabled ? targetCommand.motor3_speed : 0;

    if (actualSpeed[i] < targetSpeed)
    {
      int tmp = actualSpeed[i] + ACCEL_LIMIT;
      if (tmp > targetSpeed)
        tmp = targetSpeed;
      actualSpeed[i] = (int8_t)tmp;
    }
    else if (actualSpeed[i] > targetSpeed)
    {
      int tmp = actualSpeed[i] - ACCEL_LIMIT;
      if (tmp < targetSpeed)
        tmp = targetSpeed;
      actualSpeed[i] = (int8_t)tmp;
    }
  }

  // Update motor outputs
  updateMotors();

  delay(20); // 50Hz control loop
}

// ============== MOTOR CONTROL FUNCTIONS ==============
void setupMotors()
{
  Serial.println("Initializing motors...");

  // Configure motor pins
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);
  pinMode(MOTOR3_IN1, OUTPUT);
  pinMode(MOTOR3_IN2, OUTPUT);

  // Setup PWM channels
  ledcSetup(PWM_CHANNEL_1, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CHANNEL_2, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CHANNEL_3, PWM_FREQ, PWM_RESOLUTION);

  ledcAttachPin(MOTOR1_EN, PWM_CHANNEL_1);
  ledcAttachPin(MOTOR2_EN, PWM_CHANNEL_2);
  ledcAttachPin(MOTOR3_EN, PWM_CHANNEL_3);

  // Start with motors stopped
  emergencyStop();

  Serial.println("Motors initialized");
}

void setupESPNow()
{
  Serial.println("Initializing ESP-NOW...");

  WiFi.mode(WIFI_STA);

  // Print MAC address
  uint8_t myMac[6];
  esp_read_mac(myMac, ESP_MAC_WIFI_STA);
  Serial.print("My MAC Address: ");
  for (int i = 0; i < 6; i++)
  {
    Serial.printf("%02X", myMac[i]);
    if (i < 5)
      Serial.print(":");
  }
  Serial.println();

  if (!initESPNowWithRetry())
  {
    Serial.println("ERROR: ESP-NOW initialization failed after retries!");
    espNowReady = false;
    return;
  }

  // Register receive callback
  esp_now_register_recv_cb(onDataReceived);

  if (!addPeerWithRetry(bridgeMacAddress))
  {
    Serial.println("ERROR: Failed to add peer after retries");
    espNowReady = false;
    return;
  }

  espNowReady = true;
  Serial.println("ESP-NOW initialized");
}

bool initESPNowWithRetry()
{
  for (int i = 0; i < ESPNOW_INIT_RETRIES; i++)
  {
    if (esp_now_init() == ESP_OK)
    {
      return true;
    }
    Serial.printf("WARN: ESP-NOW init failed (attempt %d)\n", i + 1);
    delay(ESPNOW_RETRY_DELAY_MS);
  }
  return false;
}

bool addPeerWithRetry(const uint8_t *peerMac)
{
  if (peerMac == nullptr)
    return false;

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, peerMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  for (int i = 0; i < PEER_ADD_RETRIES; i++)
  {
    if (esp_now_add_peer(&peerInfo) == ESP_OK)
    {
      return true;
    }
    Serial.printf("WARN: Add peer failed (attempt %d)\n", i + 1);
    delay(ESPNOW_RETRY_DELAY_MS);
  }
  return false;
}

void updateMotors()
{
  setMotorSpeed(1, actualSpeed[0]);
  setMotorSpeed(2, actualSpeed[1]);
  setMotorSpeed(3, actualSpeed[2]);
}

void setMotorSpeed(uint8_t motorNum, int8_t speed)
{
  // Clamp speed to valid range
  speed = constrain(speed, -100, 100);

  uint8_t pwmValue = map(abs(speed), 0, 100, 0, 255);
  bool forward = speed >= 0;

  switch (motorNum)
  {
  case 1:
    digitalWrite(MOTOR1_IN1, forward ? HIGH : LOW);
    digitalWrite(MOTOR1_IN2, forward ? LOW : HIGH);
    ledcWrite(PWM_CHANNEL_1, pwmValue);
    break;
  case 2:
    digitalWrite(MOTOR2_IN1, forward ? HIGH : LOW);
    digitalWrite(MOTOR2_IN2, forward ? LOW : HIGH);
    ledcWrite(PWM_CHANNEL_2, pwmValue);
    break;
  case 3:
    digitalWrite(MOTOR3_IN1, forward ? HIGH : LOW);
    digitalWrite(MOTOR3_IN2, forward ? LOW : HIGH);
    ledcWrite(PWM_CHANNEL_3, pwmValue);
    break;
  }
}

void emergencyStop()
{
  // Stop all motors immediately
  ledcWrite(PWM_CHANNEL_1, 0);
  ledcWrite(PWM_CHANNEL_2, 0);
  ledcWrite(PWM_CHANNEL_3, 0);
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
  digitalWrite(MOTOR3_IN1, LOW);
  digitalWrite(MOTOR3_IN2, LOW);

  actualSpeed[0] = actualSpeed[1] = actualSpeed[2] = 0;
  targetCommand.motor1_speed = 0;
  targetCommand.motor2_speed = 0;
  targetCommand.motor3_speed = 0;
}

// ============== ESP-NOW CALLBACK ==============
void onDataReceived(const uint8_t *mac, const uint8_t *data, int len)
{
  if (data == nullptr || len != sizeof(motor_command))
  {
    Serial.printf("WARNING: Invalid packet (len=%d)\n", len);
    return;
  }

  memcpy(&targetCommand, data, sizeof(motor_command));
  lastCommandTime = millis();

  // Check emergency stop flag
  if (targetCommand.flags & 0x01)
  {
    Serial.println("EMERGENCY STOP RECEIVED");
    emergencyStop();
    motorsEnabled = false;
    return;
  }

  // Set motor enable state based on flag
  motorsEnabled = (targetCommand.flags & 0x02) != 0;

  // Debug output (optional - comment out for performance)
  Serial.printf("CMD: M1=%d M2=%d M3=%d Flags=0x%02X\n",
                targetCommand.motor1_speed,
                targetCommand.motor2_speed,
                targetCommand.motor3_speed,
                targetCommand.flags);
}