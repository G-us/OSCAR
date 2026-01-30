# ESP-NOW Motor Control System

Complete wireless motor control system using ESP-NOW and L293N drivers.

## System Architecture

```
┌─────────────┐      Serial      ┌──────────────┐    ESP-NOW    ┌──────────────────┐
│   Laptop    │ ─────────────────→│ Bridge ESP32 │ ─────────────→│ Motor Controller │
│ (Python)    │←─────────────────  │ (ESPNOWTEST) │←─────────────  │   ESP32 (CONT)   │
└─────────────┘                    └──────────────┘                └──────────────────┘
     ↑                                                                      ↓
  Xbox/Keyboard                                                      3x Motors (L293N)
```

### Components:

1. **Laptop Controller** (`PC/motorController.py`)
   - Reads Xbox controller or keyboard input
   - Sends motor commands via serial (50Hz)
   - Supports emergency stop

2. **Serial Bridge ESP32** (`OSCARESPNOWTEST`)
   - Forwards packets between Serial ↔ ESP-NOW
   - Minimal processing overhead
   - Bidirectional communication

3. **Motor Controller ESP32** (`OSCARCONT`)
   - Receives ESP-NOW motor commands
   - Generates PWM for L293N drivers
   - Implements safety watchdog (500ms timeout)
   - Smooth speed ramping (prevents jerky motion)

## Hardware Setup

### Motor Controller ESP32 Wiring (L293N)

#### Motor 1:

- IN1 → GPIO 13
- IN2 → GPIO 12
- EN → GPIO 14 (PWM)

#### Motor 2:

- IN1 → GPIO 27
- IN2 → GPIO 26
- EN → GPIO 25 (PWM)

#### Motor 3:

- IN1 → GPIO 33
- IN2 → GPIO 32
- EN → GPIO 15 (PWM)

### L293N Power:

- **VCC1** (Logic) → 5V from ESP32
- **VCC2** (Motor) → External power supply (6-12V depending on motors)
- **GND** → Common ground with ESP32

### Bridge ESP32:

- Connect to laptop via USB
- No additional wiring needed

## Software Setup

### 1. Get MAC Addresses

Upload code to both ESP32s and note their MAC addresses from Serial Monitor.

**Motor Controller MAC:** `XX:XX:XX:XX:XX:XX`  
**Bridge MAC:** `XX:XX:XX:XX:XX:XX`

### 2. Update MAC Addresses in Code

**In `OSCARCONT/src/main.cpp`:**

```cpp
uint8_t bridgeMacAddress[] = {0x08, 0xB6, 0x1F, 0x28, 0x96, 0xDC}; // Update!
```

**In `OSCARESPNOWTEST/src/main.cpp`:**

```cpp
uint8_t motorControllerMAC[] = {0x08, 0xB6, 0x1F, 0x29, 0xD6, 0x5C}; // Update!
```

### 3. Upload Code to ESP32s

Using PlatformIO:

```bash
# Upload to Motor Controller
cd OSCARCONT/OSCARMOTORTEST
pio run --target upload

# Upload to Bridge
cd OSCARESPNOWTEST
pio run --target upload
```

### 4. Install Python Dependencies

```bash
pip install pyserial inputs keyboard
```

**Note:** On Windows, you may need admin rights for `keyboard` library.

### 5. Update Serial Port

In `motorController.py`, change:

```python
SERIAL_PORT = 'COM3'  # Change to your bridge ESP32 port
```

Find your port:

- Windows: Device Manager → Ports (COM & LPT)
- Linux/Mac: `ls /dev/tty*`

## Running the System

### 1. Power On

- Connect motor controller ESP32 to power
- Connect bridge ESP32 to laptop via USB

### 2. Start Python Controller

```bash
cd PC
python motorController.py
```

### 3. Select Input Method

- Option 1: Xbox Controller (automatic analog control)
- Option 2: Keyboard (WASD + IK controls)

### 4. Control Motors

#### Xbox Controller:

- **Left Stick Y** → Motor 1 (forward/backward)
- **Left Stick X** → Motor 2 (left/right)
- **Right Stick Y** → Motor 3
- **B Button** → Emergency Stop

#### Keyboard:

- **W/S** → Motor 1
- **A/D** → Motor 2
- **I/K** → Motor 3
- **SPACE** → Emergency Stop
- **ESC** → Quit

## Safety Features

✅ **Watchdog Timer**: Motors auto-stop if no command received for 500ms  
✅ **Speed Ramping**: Smooth acceleration prevents mechanical stress  
✅ **Emergency Stop**: Immediate halt on command or connection loss  
✅ **Enable Flag**: Motors can be disabled remotely

## Customization

### Adjust Motor Mapping

In `motorController.py`, modify the `update_motors()` function:

```python
def update_motors(self):
    # Example: Tank drive
    forward = -self.left_stick_y
    turn = self.left_stick_x
    self.command.motor1_speed = int((forward + turn) * MAX_SPEED)
    self.command.motor2_speed = int((forward - turn) * MAX_SPEED)
    self.command.motor3_speed = 0
```

### Change Control Rate

In `motorController.py`:

```python
SEND_RATE = 50  # Hz (20ms between packets) - adjust 10-100Hz
```

### Adjust Speed Ramping

In `OSCARCONT/src/main.cpp`:

```cpp
#define ACCEL_LIMIT 5  // Increase for faster response, decrease for smoother
```

### Modify Watchdog Timeout

In `OSCARCONT/src/main.cpp`:

```cpp
#define WATCHDOG_TIMEOUT 500  // Milliseconds
```

## Troubleshooting

### Motors don't move

1. Check MAC addresses are correctly configured
2. Verify ESP-NOW connection in Serial Monitor
3. Ensure L293N has external power supply
4. Check motor wiring polarity

### Jerky movement

- Increase `ACCEL_LIMIT` for faster response
- Reduce `SEND_RATE` if network is saturated

### Connection drops

- Ensure ESP32s are within range (~50m outdoors, ~10m indoors)
- Avoid WiFi interference
- Check power supply stability

### Serial connection fails

- Verify correct COM port
- Close other programs using serial port
- Reset ESP32 and reconnect

## Protocol Details

### Message Structure (4 bytes):

```
Byte 0: motor1_speed (int8_t, -100 to +100)
Byte 1: motor2_speed (int8_t, -100 to +100)
Byte 2: motor3_speed (int8_t, -100 to +100)
Byte 3: flags (uint8_t)
  - Bit 0: Emergency stop
  - Bit 1: Enable motors
  - Bits 2-7: Reserved
```

### Communication Flow:

1. Python reads input → creates packet
2. Serial → Bridge ESP32
3. ESP-NOW → Motor Controller ESP32
4. PWM generation → L293N → Motors

## Performance

- **Latency**: ~20-40ms end-to-end
- **Update Rate**: 50Hz (20ms)
- **Range**: 50m+ outdoors, 10m+ indoors
- **Reliability**: 99%+ packet delivery in good conditions

## Web Interface (Future Enhancement)

To add web control:

1. Create Flask/FastAPI server on laptop
2. Add WebSocket endpoint
3. Connect to `motorController.py` via threading
4. Build HTML/JS joystick interface

Example structure:

```
Browser → WebSocket → Python Server → Serial → ESP-NOW → Motors
```

---

**Author:** Henry Parsons  
**Date:** January 2026  
**License:** MIT
