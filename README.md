# Autonomous Line-Following Robot — Q-Learning

A physical omnidirectional robot that learns to follow a line using Q-learning, trained entirely in a custom-built simulator and then deployed to hardware with no retraining.

**Hardware:** Raspberry Pi 5 · 4× DC motors with Mecanum wheels · L298N H-bridge drivers · USB camera · MPU-6050 IMU  
**Stack:** Python · NumPy · OpenCV · PyGame · gpiozero

---

## How It Works

### 1. Simulation (`robot.py`)

A PyGame environment trains the agent before any hardware is involved. The simulator deliberately avoids perfect physics — it includes inertia, vision latency, sensor dropout, and a 1% chance of a frame being lost — so the learned policy transfers to the real world without retraining.

**Procedural map generation.** Every episode uses a new randomly deformed circular track (sinusoidal radius perturbations, 2–4 waves). This prevents the agent from memorising a fixed route and forces generalisation of lane-following behaviour.

**Vision model.** Rather than giving the agent raw (x, y) coordinates, the simulator ray-casts a 64×64 virtual camera image and passes it through the same centroid-based pipeline used on the physical robot. The image is divided into three horizontal slices (near / mid / far); the centroid of white pixels in each slice becomes part of the state. This makes the sim-to-real gap minimal.

**State space:** `(lateral_position × curve_geometry × gyro) = 4 × 6 × 3 = 72 states`  
**Action space:** 5 discrete actions — forward, pivot left/right, arc left/right  
**Q-table size:** 10 × 10 × 10 × 3 × 5 × 5 = 75,000 entries (trivially small, intentionally)

**Bellman update:**
```python
new_q = old_q + α * (reward + γ * max(Q[next_state]) - old_q)
```
α = 0.15, γ = 0.90, ε decays from 1.0 → 0.05 over ~100,000 episodes.

**Reward shaping.** The reward function combines: forward progress along the track (+10 per segment), centering error (up to ±15 depending on bucket), sharp-turn anticipation from far-sensor readings, a centering-trend bonus (+4 if error decreased), and a sustained-progress bonus for consecutive forward steps. Rewards are clipped to [−50, +50] to prevent numerical instability.

Training converges in roughly 7,000–9,000 episodes (~30 minutes). The trained Q-table is saved as `brain.npy`.

---

### 2. Physical Robot (`pi.py`)

The saved Q-table is transferred to the Raspberry Pi and loaded at boot. No retraining occurs on hardware.

**Motor deadzone correction.** DC motors with gearboxes have a static friction threshold. Below ~50% PWM the motor stalls. The `Motor` class clamps any non-zero speed to a minimum of 0.5, mapping the continuous [0, 1] range to [0.5, 1.0] for motion commands.

**Mecanum wheel mixing.** Holonomic drive allows lateral movement without chassis rotation. Wheel speeds are computed from three independent axes (`ly`, `lx`, `rx`) and normalised so no motor exceeds ±1.0:
```python
fl = (ly + lx + rx) / max_v
fr = (ly - lx - rx) / max_v
rl = (ly - lx + rx) / max_v
rr = (ly + lx - rx) / max_v
```

**Gyro calibration.** At startup, 50 samples are averaged while the robot is stationary to compute a zero-offset. This compensates for MEMS sensor drift without external calibration hardware.

**Sharp-curve speed reduction.** If the state classifier detects an upcoming sharp curve (curve states 3–4), forward speed is reduced by 35% to prevent wheel slip.

---

## Results

Tested on a black-tape track with straights, gentle curves, and sharp corners:

| Track difficulty | Laps completed / 20 attempts | Success rate |
|---|---|---|
| Simple (straights + gentle curves) | 18/20 | 90% |
| Mixed (includes sharp corners) | 16/20 | 80% |

Failure modes: sudden lighting changes that shift the binarisation threshold, and wheel slip on dusty surfaces — neither is modelled in the simulator.

The robot's physical inertia acts as a natural low-pass filter on jittery decisions, producing smoother motion than the simulator itself.

---

## Repository Structure

```
robot.py        Q-learning simulator (training)
pi.py           Raspberry Pi deployment (inference)
brain.npy       Trained Q-table (not included — run robot.py to generate)
```

---

## Running

**Train in simulation:**
```bash
pip install pygame numpy opencv-python
python robot.py
```

**Deploy to robot** (Raspberry Pi 5, Python 3.11+):
```bash
pip install gpiozero opencv-python numpy pygame
# Copy brain.npy to the Pi, then:
python pi.py
```
Press joystick button 0 to toggle between manual and autonomous mode.

---

## Limitations & Next Steps

The discrete Q-table works well for this constrained problem but scales poorly. State space grows exponentially with additional sensors. Replacing the table with a Deep Q-Network (DQN) would allow raw pixel input and handle more complex environments. Adding a LiDAR sensor would enable obstacle avoidance independent of the vision pipeline.
