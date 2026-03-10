"""
ESP-NOW Motor Controller - Laptop Side
Sends motor commands to ESP32 via Serial Bridge
Supports Xbox controller and keyboard input
"""

import serial
import struct
import time
import sys
import threading

from sympy import false

# Try to import Xbox controller support
try:
    from inputs import get_gamepad

    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False
    print("Warning: 'inputs' library not found. Xbox controller support disabled.")
    print("Install with: pip install inputs")

# Try to import keyboard support
try:
    import keyboard

    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: 'keyboard' library not found. Keyboard control disabled.")
    print("Install with: pip install keyboard")

# ============== CONFIGURATION ==============
SERIAL_PORT = 'COM4'  # Change to your ESP32 serial port
BAUD_RATE = 115200
SEND_RATE = 50  # Hz (20ms between packets)

# Motor command limits
MAX_SPEED = 100
MIN_SPEED = -100

# Deadzone for analog sticks (prevents drift)
DEADZONE = 0.15

# Stepper target tuning (command units, scaled by firmware by 10x)
STEPPER_TARGET_RATE = 8
STEPPER_TARGET_STEP = 5


# ============== MOTOR COMMAND STRUCTURE ==============
# Must match ESP32 structure: int8_t, int8_t, int16_t, uint8_t
class MotorCommand:
    def __init__(self):
        self.motor1_speed = 0
        self.motor2_speed = 0
        self.stepper_target = 0
        self.flags = 0x02  # Bit 1 = enable motors by default

    def pack(self):
        """Pack command into bytes for serial transmission"""
        # Format: 2 signed bytes + 1 signed short + 1 unsigned byte
        return struct.pack('<bbhB',
                           self.motor1_speed,
                           self.motor2_speed,
                   self.stepper_target,
                           self.flags)

    def set_emergency_stop(self):
        """Set emergency stop flag"""
        self.flags |= 0x01
        self.motor1_speed = 0
        self.motor2_speed = 0
        self.stepper_target = 0

    def cancel_emergency_stop(self):
        self.flags = 0x02

    def clamp_stepper_target(self):
        """Clamp target to int16 range to match packet format"""
        if self.stepper_target > 32767:
            self.stepper_target = 32767
        elif self.stepper_target < -32768:
            self.stepper_target = -32768

    def enable_motors(self):
        """Enable motors"""
        self.flags |= 0x02

    def disable_motors(self):
        """Disable motors"""
        self.flags &= ~0x02


# ============== XBOX CONTROLLER HANDLER ==============
class XboxController:
    def __init__(self, command):
        self.command = command
        self.running = True
        self.thread = None

        # Controller state
        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.right_stick_y = 0.0
        self.left_trigger = 0.0
        self.right_trigger = 0.0

    def apply_deadzone(self, value):
        """Apply deadzone to analog input"""
        if abs(value) < DEADZONE:
            return 0.0
        # Rescale to make deadzone smooth
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)

    def process_event(self, event):
        """Process gamepad events"""
        # Left stick (Motor 1 & 2)
        if event.code == 'ABS_X':
            self.left_stick_x = self.apply_deadzone(event.state / 32768.0)
        elif event.code == 'ABS_Y':
            self.left_stick_y = self.apply_deadzone(event.state / 32768.0)

        # Right stick (Stepper target)
        elif event.code == 'ABS_RX':
            self.right_stick_x = self.apply_deadzone(event.state / 32768.0)
        elif event.code == 'ABS_RY':
            self.right_stick_y = self.apply_deadzone(event.state / 32768.0)

        # Triggers
        elif event.code == 'ABS_Z':  # Left trigger
            self.left_trigger = event.state / 255.0
        elif event.code == 'ABS_RZ':  # Right trigger
            self.right_trigger = event.state / 255.0

        # Emergency stop on button press (e.g., B button)
        elif event.code == 'BTN_EAST' and event.state == 1:
            print("EMERGENCY STOP!")
            self.command.set_emergency_stop()

    def update_motors(self):
        """Convert controller input to motor commands (differential drive)"""
        # Differential drive: combine forward/backward with rotation
        # Left stick Y -> forward/backward (both motors same direction)
        # Left stick X -> rotation (motors opposite directions)

        forward = -self.left_stick_y  # Forward is positive
        rotation = self.left_stick_x  # Right is positive

        # Combine forward and rotation
        left_motor = forward + rotation
        right_motor = forward - rotation

        # Clamp to motor limits
        left_motor = max(min(left_motor, 1.0), -1.0)
        right_motor = max(min(right_motor, 1.0), -1.0)

        # Scale to motor speed range
        self.command.motor1_speed = int(left_motor * MAX_SPEED)
        self.command.motor2_speed = int(right_motor * MAX_SPEED)

        # Adjust stepper target with right stick Y (position increments)
        if abs(self.right_stick_y) > 0.05:
            delta_units = int(self.right_stick_y * STEPPER_TARGET_RATE)
            self.command.stepper_target = self.command.stepper_target + delta_units
            self.command.clamp_stepper_target()

    def run(self):
        """Main gamepad reading loop"""
        print("Xbox controller active. Press Ctrl+C to stop.")
        print("Controls:")
        print("  Left Stick Y  -> Motor 1")
        print("  Left Stick X  -> Motor 2")
        print("  Right Stick Y -> Stepper Target")
        print("  B Button      -> Emergency Stop")

        try:
            while self.running:
                events = get_gamepad()
                for event in events:
                    self.process_event(event)
                self.update_motors()
        except Exception as e:
            print(f"Controller error: {e}")

    def start(self):
        """Start controller thread"""
        self.thread = threading.Thread(target = self.run, daemon = True)
        self.thread.start()

    def stop(self):
        """Stop controller thread"""
        self.running = False


# ============== KEYBOARD CONTROLLER ==============
class KeyboardController:
    def __init__(self, command):
        self.command = command
        self.running = True
        self.rotateSpeed = 75

        # Speed increments
        self.motor1 = 0
        self.motor2 = 0
        self.stepper_step = STEPPER_TARGET_STEP
        self.slowRotate = False

    def setup_hotkeys(self):
        """Setup keyboard shortcuts"""
        print("\nKeyboard controls:")
        print("  W -> Forward")
        print("  S -> Backward")
        print("  A -> Rotate Left")
        print("  D -> Rotate Right")
        print("  R -> Stepper +")
        print("  F -> Stepper -")
        print("  T -> Stepper to 0")
        print("  SHIFT -> Slow Rotate")
        print("  SPACE -> Emergency Stop")
        print("  ESC -> Quit")

        # Forward/Backward
        keyboard.on_press_key('w', lambda _: self.set_both_motors(-75, -75))
        keyboard.on_press_key('s', lambda _: self.set_both_motors(75, 75))

        # Rotation
        keyboard.on_press_key('a', lambda _: self.set_both_motors(-self.rotateSpeed, self.rotateSpeed))
        keyboard.on_press_key('d', lambda _: self.set_both_motors(self.rotateSpeed, -self.rotateSpeed))
        keyboard.on_press_key('shift', lambda _: self.setSlowRotate(True))
        keyboard.on_release_key('shift', lambda _: self.setSlowRotate(False))

        keyboard.on_press_key('space', lambda _: self.emergency_stop())
        keyboard.on_press_key('z', lambda _: self.cancel_emergency_stop())

        # Stepper target increments
        keyboard.on_press_key('r', lambda _: self.adjust_stepper(self.stepper_step))
        keyboard.on_press_key('f', lambda _: self.adjust_stepper(-self.stepper_step))
        keyboard.on_press_key('t', lambda _: self.SetStepperAbsolute(0))

        # Release keys to stop
        keyboard.on_release_key('w', lambda _: self.set_both_motors(0, 0))
        keyboard.on_release_key('s', lambda _: self.set_both_motors(0, 0))
        keyboard.on_release_key('a', lambda _: self.set_both_motors(0, 0))
        keyboard.on_release_key('d', lambda _: self.set_both_motors(0, 0))

    def set_both_motors(self, left, right):
        """Set both motor speeds for differential drive"""
        self.command.motor1_speed = left
        self.command.motor2_speed = right

    def emergency_stop(self):
        """Emergency stop"""
        print("EMERGENCY STOP!")
        self.command.set_emergency_stop()

    def cancel_emergency_stop(self):
        """Cancel emergency stop"""
        self.command.cancel_emergency_stop()

    def adjust_stepper(self, delta_units):
        """Adjust stepper target position in command units"""
        self.command.stepper_target = self.command.stepper_target + delta_units
        self.command.clamp_stepper_target()

    def setSlowRotate(self, value):
        """Set slow rotation mode"""
        self.slowRotate = value
        if self.slowRotate:
            print("Slow rotation enabled")
            self.rotateSpeed = 40
        else:
            print("Slow rotation disabled")
            self.rotateSpeed = 75

    def SetStepperAbsolute(self, position):
        self.command.stepper_target = position
        self.command.clamp_stepper_target()
        print(f"Stepper target set to {position}")

# ============== MAIN CONTROLLER ==============
class MotorControllerApp:
    def __init__(self):
        self.command = MotorCommand()
        self.serial_conn = None
        self.running = True

        # Input handlers
        self.xbox_controller = None
        self.keyboard_controller = None

    def connect_serial(self):
        """Connect to ESP32 serial bridge"""
        print(f"Connecting to {SERIAL_PORT}...")
        try:
            self.serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout = 0.1)
            time.sleep(2)  # Wait for ESP32 to reset
            print("Connected!")

            # Clear any startup messages
            while self.serial_conn.in_waiting:
                print(self.serial_conn.readline().decode('utf-8', errors = 'ignore').strip())

            return True
        except Exception as e:
            print(f"Error connecting: {e}")
            return False

    def send_command(self):
        """Send motor command to ESP32"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                packet = self.command.pack()
                self.serial_conn.write(packet)
                self.serial_conn.flush()
            except Exception as e:
                print(f"Send error: {e}")

    def run(self):
        """Main control loop"""
        if not self.connect_serial():
            return

        # Choose input method
        if GAMEPAD_AVAILABLE:
            print("\nInput method:")
            print("1. Xbox Controller")
            print("2. Keyboard")
            choice = input("Select (1 or 2): ").strip()

            if choice == '1':
                self.xbox_controller = XboxController(self.command)
                self.xbox_controller.start()
            elif choice == '2' and KEYBOARD_AVAILABLE:
                self.keyboard_controller = KeyboardController(self.command)
                self.keyboard_controller.setup_hotkeys()
        elif KEYBOARD_AVAILABLE:
            self.keyboard_controller = KeyboardController(self.command)
            self.keyboard_controller.setup_hotkeys()
        else:
            print("No input methods available!")
            return

        print(f"\nSending commands at {SEND_RATE}Hz. Press Ctrl+C to stop.")

        try:
            interval = 1.0 / SEND_RATE
            last_send = time.time()

            while self.running:
                current_time = time.time()

                if current_time - last_send >= interval:
                    serialResponse = self.serial_conn.readline().decode('utf-8', errors = 'ignore').strip()
                    print(serialResponse)
                    if not (serialResponse == "ERROR: ESP-NOW send failed"):
                        self.send_command()
                        last_send = current_time
                        # Optional: Print status
                        print(
                            f"M1:{self.command.motor1_speed:4d} "
                            f"M2:{self.command.motor2_speed:4d} "
                            f"Step:{self.command.stepper_target:6d}",
                            end = '\r'
                        )
                    else:
                        print("ESP-NOW send failed")

                time.sleep(0.001)  # Small sleep to prevent CPU hogging

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Send stop command
            self.command.set_emergency_stop()
            self.send_command()
            time.sleep(0.1)

            if self.xbox_controller:
                self.xbox_controller.stop()

            if self.serial_conn:
                self.serial_conn.close()

            print("Disconnected.")


# ============== ENTRY POINT ==============
if __name__ == "__main__":
    print("=== ESP-NOW Motor Controller ===")
    app = MotorControllerApp()
    app.run()
