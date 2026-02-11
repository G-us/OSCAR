/*
 * ESP32 Serial Bridge for ESP-NOW Communication
 * Forwards packets between laptop (via Serial) and motor controller (via ESP-NOW)
 * Bidirectional: Serial <-> ESP-NOW
 */

#include <esp_now.h>
#include <WiFi.h>
#include <Arduino.h>

// ============== CONFIGURATION ==============
// MAC address of motor controller ESP32 (update with actual MAC)
uint8_t motorControllerMAC[] = {0x08, 0xB6, 0x1F, 0x28, 0x96, 0xDC};

// Resilience Configuration
#define ESPNOW_INIT_RETRIES 5
#define ESPNOW_RETRY_DELAY_MS 250
#define PEER_ADD_RETRIES 3
#define SERIAL_PACKET_TIMEOUT_MS 20

// Message structure - must match motor controller and Python
typedef struct __attribute__((packed)) motor_command
{
  int8_t motor1_speed;   // -100 to +100
  int8_t motor2_speed;   // -100 to +100
  int16_t stepper_target; // Target position (scaled by controller)
  uint8_t flags;         // Control flags
} motor_command;

// ============== STATE VARIABLES ==============
motor_command outgoingCommand;
motor_command incomingCommand;
unsigned long lastSerialReceive = 0;
unsigned long lastESPNowReceive = 0;
bool espNowReady = false;

// ============== FUNCTION PROTOTYPES ==============
void setupESPNow();
bool initESPNowWithRetry();
bool addPeerWithRetry(const uint8_t *peerMac);
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status);
void onDataReceived(const uint8_t *mac, const uint8_t *data, int len);
void processSerialData();

void setup()
{
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 Serial Bridge Starting ===");

  setupESPNow();

  Serial.println("=== Bridge Ready - Awaiting Commands ===");
}

void loop()
{
  if (!espNowReady)
  {
    static unsigned long lastRetry = 0;
    if (millis() - lastRetry > 2000)
    {
      lastRetry = millis();
      setupESPNow();
    }
    delay(5);
    return;
  }

  // Check for data from laptop via Serial
  processSerialData();

  delay(1); // Small delay to prevent watchdog issues
}

// ============== ESP-NOW SETUP ==============
void setupESPNow()
{
  Serial.println("Initializing ESP-NOW bridge...");

  WiFi.mode(WIFI_STA);

  // Print this device's MAC address
  uint8_t myMac[6];
  esp_read_mac(myMac, ESP_MAC_WIFI_STA);
  Serial.print("Bridge MAC Address: ");
  for (int i = 0; i < 6; i++)
  {
    Serial.printf("%02X", myMac[i]);
    if (i < 5)
      Serial.print(":");
  }
  Serial.println();
  Serial.print("Motor Controller MAC: ");
  for (int i = 0; i < 6; i++)
  {
    Serial.printf("%02X", motorControllerMAC[i]);
    if (i < 5)
      Serial.print(":");
  }
  Serial.println();

  // Initialize ESP-NOW
  if (!initESPNowWithRetry())
  {
    Serial.println("ERROR: ESP-NOW initialization failed after retries!");
    espNowReady = false;
    return;
  }

  // Register callbacks
  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataReceived);

  if (!addPeerWithRetry(motorControllerMAC))
  {
    Serial.println("ERROR: Failed to add peer after retries");
    espNowReady = false;
    return;
  }

  espNowReady = true;
  Serial.println("ESP-NOW bridge initialized");
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

// ============== SERIAL PROCESSING ==============
void processSerialData()
{
  // Check if we have enough bytes for a complete packet
  if (Serial.available() >= sizeof(motor_command))
  {
    // Read the packet
    uint8_t buffer[sizeof(motor_command)];
    size_t bytesRead = Serial.readBytes(buffer, sizeof(motor_command));

    if (bytesRead != sizeof(motor_command))
    {
      Serial.println("WARNING: Short serial read, dropping packet");
      return;
    }

    memcpy(&outgoingCommand, buffer, sizeof(motor_command));
    lastSerialReceive = millis();

    // Forward to motor controller via ESP-NOW
    esp_err_t result = esp_now_send(motorControllerMAC, (uint8_t *)&outgoingCommand, sizeof(motor_command));

    if (result != ESP_OK)
    {
      Serial.println("ERROR: ESP-NOW send failed");
    }
    // Optional: Echo for debugging (comment out for production)
    Serial.printf("Forwarded: M1=%d M2=%d Stepper=%d\n",
                  outgoingCommand.motor1_speed,
                  outgoingCommand.motor2_speed,
                  outgoingCommand.stepper_target);
  }
}

// ============== ESP-NOW CALLBACKS ==============
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status)
{
  // Optional: Send status back to laptop
  if (status != ESP_NOW_SEND_SUCCESS)
  {
    // Could send error packet back to laptop here
  }
}

void onDataReceived(const uint8_t *mac, const uint8_t *data, int len)
{
  // Receive data from motor controller (telemetry, acks, etc.)
  if (data == nullptr || len != sizeof(motor_command))
  {
    Serial.printf("WARNING: Invalid ESP-NOW packet (len=%d)\n", len);
    return;
  }

  memcpy(&incomingCommand, data, len);
  lastESPNowReceive = millis();

  // Forward to laptop via Serial
  Serial.write((uint8_t *)&incomingCommand, sizeof(motor_command));
  Serial.flush();
}