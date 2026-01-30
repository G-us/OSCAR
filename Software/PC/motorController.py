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
SERIAL_PORT = 'COM3'  # Change to your ESP32 serial port
BAUD_RATE = 115200
SEND_RATE = 50  # Hz (20ms between packets)

# Motor command limits
MAX_SPEED = 100
MIN_SPEED = -100

# Deadzone for analog sticks (prevents drift)
DEADZONE = 0.15

# ============== MOTOR COMMAND STRUCTURE ==============
# Must match ESP32 structure: int8_t, int8_t, int8_t, uint8_t
class MotorCommand:
    def __init__(self):
        self.motor1_speed = 0
        self.motor2_speed = 0
        self.motor3_speed = 0
        self.flags = 0x02  # Bit 1 = enable motors by default
    
    def pack(self):
        """Pack command into bytes for serial transmission"""
        # Format: 3 signed bytes + 1 unsigned byte
        return struct.pack('bbbB', 
                          self.motor1_speed, 
                          self.motor2_speed, 
                          self.motor3_speed, 
                          self.flags)
    
    def set_emergency_stop(self):
        """Set emergency stop flag"""
        self.flags |= 0x01
        self.motor1_speed = 0
        self.motor2_speed = 0
        self.motor3_speed = 0
    
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
        
        # Right stick (Motor 3)
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
        """Convert controller input to motor commands"""
        # Example mapping (customize for your robot):
        # Left stick Y -> Motor 1 (forward/backward)
        # Left stick X -> Motor 2 (left/right)
        # Right stick Y -> Motor 3
        
        self.command.motor1_speed = int(-self.left_stick_y * MAX_SPEED)
        self.command.motor2_speed = int(self.left_stick_x * MAX_SPEED)
        self.command.motor3_speed = int(-self.right_stick_y * MAX_SPEED)
    
    def run(self):
        """Main gamepad reading loop"""
        print("Xbox controller active. Press Ctrl+C to stop.")
        print("Controls:")
        print("  Left Stick Y  -> Motor 1")
        print("  Left Stick X  -> Motor 2")
        print("  Right Stick Y -> Motor 3")
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
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop controller thread"""
        self.running = False

# ============== KEYBOARD CONTROLLER ==============
class KeyboardController:
    def __init__(self, command):
        self.command = command
        self.running = True
        
        # Speed increments
        self.motor1 = 0
        self.motor2 = 0
        self.motor3 = 0
    
    def setup_hotkeys(self):
        """Setup keyboard shortcuts"""
        print("\nKeyboard controls:")
        print("  W/S -> Motor 1 (forward/backward)")
        print("  A/D -> Motor 2 (left/right)")
        print("  I/K -> Motor 3 (up/down)")
        print("  SPACE -> Emergency Stop")
        print("  ESC -> Quit")
        
        keyboard.on_press_key('w', lambda _: self.set_motor(1, 75))
        keyboard.on_press_key('s', lambda _: self.set_motor(1, -75))
        keyboard.on_press_key('a', lambda _: self.set_motor(2, -75))
        keyboard.on_press_key('d', lambda _: self.set_motor(2, 75))
        keyboard.on_press_key('i', lambda _: self.set_motor(3, 75))
        keyboard.on_press_key('k', lambda _: self.set_motor(3, -75))
        keyboard.on_press_key('space', lambda _: self.emergency_stop())
        
        # Release keys to stop
        keyboard.on_release_key('w', lambda _: self.set_motor(1, 0))
        keyboard.on_release_key('s', lambda _: self.set_motor(1, 0))
        keyboard.on_release_key('a', lambda _: self.set_motor(2, 0))
        keyboard.on_release_key('d', lambda _: self.set_motor(2, 0))
        keyboard.on_release_key('i', lambda _: self.set_motor(3, 0))
        keyboard.on_release_key('k', lambda _: self.set_motor(3, 0))
    
    def set_motor(self, motor_num, speed):
        """Set motor speed"""
        if motor_num == 1:
            self.command.motor1_speed = speed
        elif motor_num == 2:
            self.command.motor2_speed = speed
        elif motor_num == 3:
            self.command.motor3_speed = speed
    
    def emergency_stop(self):
        """Emergency stop"""
        print("EMERGENCY STOP!")
        self.command.set_emergency_stop()

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
            self.serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2)  # Wait for ESP32 to reset
            print("Connected!")
            
            # Clear any startup messages
            while self.serial_conn.in_waiting:
                print(self.serial_conn.readline().decode('utf-8', errors='ignore').strip())
            
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
                    self.send_command()
                    last_send = current_time
                    
                    # Optional: Print status
                    # print(f"M1:{self.command.motor1_speed:4d} M2:{self.command.motor2_speed:4d} M3:{self.command.motor3_speed:4d}", end='\r')
                
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
