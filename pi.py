import pygame
from gpiozero import PWMOutputDevice, Servo
import time
import smbus
import math
import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Motor GPIO pins: (ENA, IN1, IN2)
FL_ENA, FL_IN1, FL_IN2 = 22,  9, 12   # Front-left
FR_ENA, FR_IN1, FR_IN2 = 14, 18, 15   # Front-right
RL_ENA, RL_IN1, RL_IN2 =  4, 17, 27   # Rear-left
RR_ENA, RR_IN1, RR_IN2 = 23, 24, 25   # Rear-right
SERVO_PIN = 8

# Camera resolution
WIDTH, HEIGHT = 320, 240
CENTER_X = WIDTH // 2

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# ==========================================
# Q-TABLE (AI BRAIN)
# ==========================================
BRAIN_FILE = "brain_v10_short_sight.npy"
q_table = None

print(f"Loading: {BRAIN_FILE}...")
if os.path.exists(BRAIN_FILE):
    try:
        data = np.load(BRAIN_FILE, allow_pickle=True).item()
        q_table = data['q_table']
        print(">>> Brain loaded successfully <<<")
    except Exception as e:
        print(f"!!! Error reading brain file: {e} !!!")
else:
    print("!!! Brain file not found !!!")

# Physical action mapping: (forward, strafe, rotation)
# Values are PWM duty cycles [0.0 – 1.0]. Minimum 0.5 to overcome motor deadzone.
AI_ACTIONS = {
    0: (0.70, 0,    0  ),   # Fast forward
    1: (0.50, 0,    0  ),   # Slow forward (deadzone minimum)
    2: (0.50, 0, -0.5 ),   # Arc left
    3: (0.50, 0,  0.5 ),   # Arc right
    4: (0,    0, -0.6 ),   # Pivot left  (60% power)
    5: (0,    0,  0.6 ),   # Pivot right
}
ACTION_NAMES = ["FAST", "SLOW", "ARC_L", "ARC_R", "PIVOT_L", "PIVOT_R"]

# ==========================================
# VISION — SLICE-BASED CENTROID DETECTION
# ==========================================

def get_state_from_frame(frame_binary, gyro_z):
    """
    Divide the binary frame into three horizontal slices (bottom / mid / top).
    Compute the centroid of white pixels in each slice to detect lane geometry.

    Returns:
        state   – (lateral, curve, gyro) tuple used as Q-table key
        centers – (cx_bot, cx_mid, cx_top) pixel x-coordinates (None if lost)
    """
    h, w = frame_binary.shape
    slice_h = h // 3

    slices = [
        frame_binary[2 * slice_h : h,          :],  # Bottom (nearest)
        frame_binary[    slice_h : 2 * slice_h, :],  # Middle
        frame_binary[0           :     slice_h, :],  # Top (farthest)
    ]

    centers = []
    for s in slices:
        _, xs = np.where(s > 0)
        centers.append(int(np.mean(xs)) if len(xs) >= 30 else None)

    cx_bot, cx_mid, cx_top = centers

    # --- Lateral position (4 states) ---
    if cx_bot is None:
        lat = 3  # Line lost
    else:
        offset = cx_bot - CENTER_X
        if   offset < -35: lat = 0   # Left
        elif offset >  35: lat = 2   # Right
        else:              lat = 1   # Center

    # --- Curve geometry (6 states) ---
    curve = 5  # Default: straight ahead
    if cx_bot is not None:
        if cx_mid is not None:
            slope = cx_mid - cx_bot
            if cx_top is None:
                # Only two reference points available
                if   cx_mid < 20:          curve = 3  # Hitting left wall
                elif cx_mid > w - 20:      curve = 4  # Hitting right wall
                elif slope < -20:          curve = 3
                elif slope >  20:          curve = 4
                else:                      curve = 5
            else:
                # Full three-point geometry
                total_delta = cx_top - cx_bot
                if   total_delta < -35: curve = 0  # Sharp left
                elif total_delta >  35: curve = 2  # Sharp right
                else:                   curve = 1  # Gentle / straight
        else:
            # Only bottom centroid visible
            if   cx_bot < CENTER_X - 30: curve = 3
            elif cx_bot > CENTER_X + 30: curve = 4
            else:                         curve = 1

    # --- Gyroscope (3 states) ---
    if   gyro_z < -3000: gyro = 0  # Rotating left
    elif gyro_z >  3000: gyro = 2  # Rotating right
    else:                gyro = 1  # Stable

    return (lat, curve, gyro), (cx_bot, cx_mid, cx_top)


# ==========================================
# HARDWARE — MPU6050 GYROSCOPE
# ==========================================

try:
    bus = smbus.SMBus(1)
except Exception:
    print("I2C init failed — check wiring.")
    exit()


def MPU_Init():
    """Configure MPU-6050 for ±500 °/s gyro range via I2C."""
    bus.write_byte_data(0x68, 0x19, 7)    # Sample rate divider
    bus.write_byte_data(0x68, 0x6B, 1)    # Wake up
    bus.write_byte_data(0x68, 0x1A, 0)    # No DLPF
    bus.write_byte_data(0x68, 0x1B, 0x08) # Gyro FS = ±500 °/s
    bus.write_byte_data(0x68, 0x38, 1)    # Enable data-ready interrupt


def read_raw_data(addr):
    """Read a signed 16-bit value from two consecutive I2C registers."""
    try:
        high = bus.read_byte_data(0x68, addr)
        low  = bus.read_byte_data(0x68, addr + 1)
        val  = (high << 8) | low
        if val > 32768:
            val -= 65536
        return val
    except Exception:
        return 0


# ==========================================
# HARDWARE — MOTOR DRIVER (L298N H-BRIDGE)
# ==========================================

class Motor:
    """
    Abstraction for one DC motor connected via an L298N H-bridge.

    PWM deadzone correction: any non-zero speed below 0.5 is clamped to 0.5
    to ensure the motor overcomes static friction.
    """

    def __init__(self, ena: int, in1: int, in2: int):
        self.ena = PWMOutputDevice(ena)
        self.in1 = PWMOutputDevice(in1)
        self.in2 = PWMOutputDevice(in2)

    def move(self, speed: float):
        """
        Set motor speed in [-1.0, 1.0].
        Positive = forward, negative = reverse, zero = brake.
        """
        speed = max(min(speed, 1.0), -1.0)

        # Deadzone correction
        if speed != 0 and abs(speed) < 0.5:
            speed = 0.5 if speed > 0 else -0.5

        self.ena.value = abs(speed)
        if speed > 0:
            self.in1.value, self.in2.value = 1, 0   # Forward
        elif speed < 0:
            self.in1.value, self.in2.value = 0, 1   # Reverse
        else:
            self.in1.value, self.in2.value = 0, 0   # Brake


# Instantiate motors
fl = Motor(FL_ENA, FL_IN1, FL_IN2)
fr = Motor(FR_ENA, FR_IN1, FR_IN2)
rl = Motor(RL_ENA, RL_IN1, RL_IN2)
rr = Motor(RR_ENA, RR_IN1, RR_IN2)
servo = Servo(SERVO_PIN)

# OpenCV threshold trackbar
def nothing(x): pass
cv2.namedWindow("Control Panel")
cv2.createTrackbar("Threshold", "Control Panel", 100, 255, nothing)

# Joystick init
pygame.init()
pygame.joystick.init()
try:
    j = pygame.joystick.Joystick(0)
    j.init()
except Exception:
    pass

# ==========================================
# GYRO CALIBRATION
# ==========================================
MPU_Init()
time.sleep(1)

gyro_z_offset = 0
for _ in range(50):
    gyro_z_offset += read_raw_data(0x47)
    time.sleep(0.005)
gyro_z_offset /= 50
print(f"Gyro calibrated. Offset: {gyro_z_offset:.1f}")

# ==========================================
# MAIN LOOP
# ==========================================
print("Ready. Minimum motor speed: 0.50 (deadzone active).")

try:
    while True:
        pygame.event.pump()
        ret, frame = cap.read()
        if not ret:
            continue

        # Image preprocessing
        thresh_val = cv2.getTrackbarPos("Threshold", "Control Panel")
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # State estimation
        gz = read_raw_data(0x47) - gyro_z_offset
        state, pts = get_state_from_frame(mask, gz)

        # Build visualization overlay
        display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if pts[0]: cv2.circle(display_img, (pts[0], 200), 5, (0,   0, 255), -1)  # Red   = bottom
        if pts[1]: cv2.circle(display_img, (pts[1], 120), 5, (255, 0,   0), -1)  # Blue  = mid
        if pts[2]: cv2.circle(display_img, (pts[2],  40), 5, (0, 255,   0), -1)  # Green = top
        if pts[0] and pts[1]:
            cv2.line(display_img, (pts[0], 200), (pts[1], 120), (0, 255, 255), 2)
        if pts[1] and pts[2]:
            cv2.line(display_img, (pts[1], 120), (pts[2],  40), (0, 255,   0), 2)

        ly = lx = rx = 0

        if j.get_button(0) and q_table is not None:
            # --- Autonomous mode ---
            action_idx = np.argmax(q_table[state])
            ly, lx, rx = AI_ACTIONS[action_idx]

            # Speed reduction on sharp curves (states 3 & 4)
            if state[1] in [3, 4]:
                ly *= 0.65

            cv2.rectangle(display_img, (0, 0), (160, 40), (0, 0, 0), -1)
            cv2.putText(display_img, f"AI: {ACTION_NAMES[action_idx]}",
                        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"\rAI: {ACTION_NAMES[action_idx]} | State: {state}", end="")
        else:
            # --- Manual mode ---
            lx = j.get_axis(0) if abs(j.get_axis(0)) > 0.1 else 0
            ly = -j.get_axis(1) if abs(j.get_axis(1)) > 0.1 else 0
            rx = 0.7 * j.get_axis(3) if abs(j.get_axis(3)) > 0.1 else 0

            cv2.rectangle(display_img, (0, 0), (160, 40), (0, 0, 0), -1)
            cv2.putText(display_img, "MANUAL",
                        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mecanum wheel mixing (holonomic drive)
        # Normalise so no wheel exceeds ±1.0
        max_v = max(abs(ly + lx + rx), abs(ly - lx - rx),
                    abs(ly - lx + rx), abs(ly + lx - rx), 1)
        fl.move((ly + lx + rx) / max_v)
        fr.move((ly - lx - rx) / max_v)
        rl.move((ly - lx + rx) / max_v)
        rr.move((ly + lx - rx) / max_v)

        cv2.imshow("Camera", frame)
        cv2.imshow("Vision", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    for motor in [fl, fr, rl, rr]:
        motor.move(0)
    print("\nRobot stopped.")
