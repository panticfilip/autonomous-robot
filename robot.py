import pygame
import numpy as np
import math
import random
import traceback
import os

# ==========================================
# CONFIGURATION
# ==========================================
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 600
SIM_WIDTH = 800
TRACK_WIDTH = 60
ROBOT_SPEED = 6
ROTATION_SPEED = 5
ROTATION_SPEED_WHILE_MOVING = 2.5
STEPS_TO_REPEAT = 2

# Simulation realism settings
VISION_LATENCY_FRAMES = 2
VISION_DROPOUT_CHANCE = 0.01

# Q-learning hyperparameters
EPISODES = 100000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
SHOW_EVERY = 100

ACTIONS = ["FORWARD", "ROT_L", "ROT_R", "FORWARD_L", "FORWARD_R"]
BUCKETS_LINE = 10
BUCKETS_GYRO = 3
BUCKETS_ACTION = len(ACTIONS)

Q_FILE = "robot_brain_random_maps_v2.npy"

# Load existing Q-table or initialize a new one
if os.path.exists(Q_FILE):
    print(f"Loading brain: {Q_FILE}")
    q_table = np.load(Q_FILE)
    epsilon = 0.3
else:
    print("Starting fresh training (Random Maps + Action Memory).")
    q_table = np.zeros((BUCKETS_LINE, BUCKETS_LINE, BUCKETS_LINE,
                        BUCKETS_GYRO, BUCKETS_ACTION, len(ACTIONS)))
    epsilon = 1.0

epsilon_decay = 0.999995
epsilon_min = 0.05

# Colors
BLACK    = (0, 0, 0);    WHITE  = (255, 255, 255); GRAY     = (50, 50, 50)
GREEN    = (0, 255, 0);  RED    = (255, 0, 0);     BLUE     = (50, 50, 255)
DARK_RED = (50, 0, 0);   YELLOW = (255, 255, 0);   ORANGE   = (255, 165, 0)


class Track:
    """Procedurally generated circular track. New layout on every episode."""

    def __init__(self):
        self.points = []
        center_x = SIM_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2

        base_radius = 230
        num_points = 350

        # Random sinusoidal deformations keep curves below ~90°
        waves = []
        for _ in range(random.randint(2, 4)):
            freq  = random.randint(2, 4)
            amp   = random.randint(15, 40)
            phase = random.uniform(0, 2 * math.pi)
            waves.append((freq, amp, phase))

        noise_strength = 2

        for i in range(num_points):
            theta = (i / num_points) * 2 * math.pi
            r = base_radius
            for freq, amp, phase in waves:
                r += amp * math.sin(freq * theta + phase)
            r = max(150, min(r, 270))
            x = center_x + int(r * math.cos(theta)) + random.uniform(-noise_strength, noise_strength)
            y = center_y + int(r * math.sin(theta)) + random.uniform(-noise_strength, noise_strength)
            self.points.append((x, y))

    def get_closest_index(self, rx, ry):
        """Return the index of the track point nearest to (rx, ry)."""
        min_dist = 1000
        closest_idx = 0
        for i, (px, py) in enumerate(self.points):
            dist = math.hypot(rx - px, ry - py)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx, min_dist


class Robot:
    """Simulated omni-directional robot with ray-cast vision and gyro."""

    def __init__(self, track):
        start_idx = 0
        sx, sy = track.points[start_idx]
        next_idx = (start_idx + 4) % len(track.points)
        nx, ny = track.points[next_idx]

        self.x = sx
        self.y = sy
        self.angle = math.degrees(math.atan2(ny - sy, nx - sx))
        self.gyro_z = 0
        self.last_discrete = [5, 5, 5]
        self.last_raw = [0.0, 0.0, 0.0]
        self.vision_frame_counter = 0
        self.consecutive_lost = 0
        self.last_action = 0
        self.steps_forward = 0
        self.last_bucket_near = 5

    def move(self, action_idx):
        rad = math.radians(self.angle)
        self.gyro_z = 0
        slip = random.uniform(0.97, 1.03)  # Small slip simulation

        if action_idx == 0:    # Forward
            self.x += ROBOT_SPEED * math.cos(rad)
            self.y += ROBOT_SPEED * math.sin(rad)
            self.steps_forward += 1
        elif action_idx == 1:  # Rotate left (pivot)
            self.angle -= ROTATION_SPEED * slip
            self.gyro_z = -1
            self.steps_forward = 0
        elif action_idx == 2:  # Rotate right (pivot)
            self.angle += ROTATION_SPEED * slip
            self.gyro_z = 1
            self.steps_forward = 0
        elif action_idx == 3:  # Forward + gentle left arc
            self.angle -= ROTATION_SPEED_WHILE_MOVING * slip
            self.x += ROBOT_SPEED * math.cos(rad)
            self.y += ROBOT_SPEED * math.sin(rad)
            self.gyro_z = -1
            self.steps_forward += 1
        elif action_idx == 4:  # Forward + gentle right arc
            self.angle += ROTATION_SPEED_WHILE_MOVING * slip
            self.x += ROBOT_SPEED * math.cos(rad)
            self.y += ROBOT_SPEED * math.sin(rad)
            self.gyro_z = 1
            self.steps_forward += 1

    def get_sensors_discrete(self, track):
        """
        Simulate three distance-based line sensors (near / mid / far).
        Returns (discrete_bucket_list, raw_error_list).
        Bucket 4 = centered; bucket 9 = line lost.
        """
        self.vision_frame_counter += 1
        if self.vision_frame_counter < VISION_LATENCY_FRAMES:
            return self.last_discrete, self.last_raw

        self.vision_frame_counter = 0
        sensor_offsets = [30, 70, 110]
        discrete_states = []
        raw_errors = []
        rad = math.radians(self.angle)

        for dist in sensor_offsets:
            sx = self.x + dist * math.cos(rad)
            sy = self.y + dist * math.sin(rad)

            closest_pt = None
            curr_min = 1000
            for px, py in track.points:
                d = math.hypot(sx - px, sy - py)
                if d < curr_min:
                    curr_min = d
                    closest_pt = (px, py)

            is_visible = False
            local_y = 0
            if curr_min < 40:
                dx = closest_pt[0] - sx
                dy = closest_pt[1] - sy
                local_y = -dx * math.sin(rad) + dy * math.cos(rad)
                local_y += random.uniform(-3, 3)
                if random.random() > VISION_DROPOUT_CHANCE:
                    is_visible = True

            if is_visible:
                raw_errors.append(local_y / (TRACK_WIDTH / 2))
                if   local_y < -28: bucket = 0
                elif local_y < -20: bucket = 1
                elif local_y < -12: bucket = 2
                elif local_y < -5:  bucket = 3
                elif local_y <  5:  bucket = 4  # Center
                elif local_y < 12:  bucket = 5
                elif local_y < 20:  bucket = 6
                elif local_y < 28:  bucket = 7
                else:               bucket = 8
                discrete_states.append(bucket)
            else:
                discrete_states.append(9)
                raw_errors.append(2.0)

        self.last_discrete = discrete_states
        self.last_raw = raw_errors
        return discrete_states, raw_errors


def get_state_tuple(discrete_sensors, gyro, last_action):
    """Pack sensor readings, gyro bucket, and last action into a Q-table key."""
    g_idx = int(gyro + 1)
    return tuple(discrete_sensors + [g_idx, last_action])


def draw_combined_vision(screen, robot, track, sensors_discrete):
    cam_x = SIM_WIDTH
    cam_y = 0
    cam_w = WINDOW_WIDTH - SIM_WIDTH
    cam_h = WINDOW_HEIGHT // 2

    pygame.draw.rect(screen, BLACK, (cam_x, cam_y, cam_w, cam_h))
    cam_cx = cam_x + cam_w // 2 - 20
    cam_cy = cam_h - 20

    robot_rad = math.radians(robot.angle)
    cos_a = math.cos(-robot_rad)
    sin_a = math.sin(-robot_rad)

    view_points = []
    for px, py in track.points:
        dist_from_robot = math.hypot(px - robot.x, py - robot.y)
        if dist_from_robot < 40 or dist_from_robot > 160:
            continue
        dx = px - robot.x
        dy = py - robot.y
        local_x =  dx * cos_a - dy * sin_a
        local_y =  dx * sin_a + dy * cos_a
        if local_x > 0:
            screen_x = cam_cx + local_y * 2.5
            screen_y = cam_cy - local_x * 2.0
            if abs(screen_x - cam_cx) < (cam_w // 2) - 40:
                view_points.append((screen_x, screen_y))

    if len(view_points) > 1:
        pygame.draw.lines(screen, WHITE, False, view_points, 20)

    row_height = cam_h // 3
    grid_width = int(cam_w * 0.75)
    col_width   = grid_width // 9
    lost_box_x  = cam_x + grid_width + 10
    lost_box_w  = (cam_w - grid_width) - 20

    font = pygame.font.SysFont('Arial', 10)
    sensor_indices = [2, 1, 0]  # Draw far → near (top → bottom)

    for row_i in range(3):
        sensor_idx  = sensor_indices[row_i]
        active_bucket = sensors_discrete[sensor_idx]
        y_pos = cam_y + row_i * row_height

        for col_i in range(9):
            x_pos = cam_x + col_i * col_width
            rect  = pygame.Rect(x_pos, y_pos, col_width, row_height)
            if col_i == active_bucket:
                pygame.draw.rect(screen, GREEN, rect, 3)
                pygame.draw.circle(screen, GREEN, rect.center, 4)
            elif col_i == 4:
                pygame.draw.rect(screen, (0, 100, 0), rect, 1)
            else:
                pygame.draw.rect(screen, GRAY, rect, 1)

        lost_rect = pygame.Rect(lost_box_x, y_pos + 10, lost_box_w, row_height - 20)
        if active_bucket == 9:
            pygame.draw.rect(screen, RED, lost_rect)
            pygame.draw.rect(screen, WHITE, lost_rect, 2)
            label = font.render("LOST", True, BLACK)
            screen.blit(label, (lost_rect.x + 5, lost_rect.y + 8))
        else:
            pygame.draw.rect(screen, DARK_RED, lost_rect)


def draw_info_panel(screen, episode, reward, epsilon, action, laps, last_action):
    panel_rect = pygame.Rect(SIM_WIDTH, WINDOW_HEIGHT // 2,
                             WINDOW_WIDTH - SIM_WIDTH, WINDOW_HEIGHT // 2)
    pygame.draw.rect(screen, (40, 40, 40), panel_rect)
    pygame.draw.line(screen, WHITE,
                     (SIM_WIDTH, WINDOW_HEIGHT // 2),
                     (WINDOW_WIDTH, WINDOW_HEIGHT // 2), 2)

    font = pygame.font.SysFont('Consolas', 18)
    lines = [
        f"EPISODE:  {episode}",
        f"REWARD:   {reward:.1f}",
        f"EPSILON:  {epsilon:.3f}",
        f"LAPS:     {laps}",
        "",
        "--- PREVIOUS ACTION ---",
        f"{ACTIONS[last_action]}",
        "",
        "--- CURRENT ACTION ---",
        f">> {ACTIONS[action]} <<"
    ]
    y = WINDOW_HEIGHT // 2 + 30
    for i, line in enumerate(lines):
        color = YELLOW if i == 6 else (GREEN if i == 9 else WHITE)
        screen.blit(font.render(line, True, color), (SIM_WIDTH + 30, y))
        y += 30


def calculate_reward(bucket_near, bucket_mid, bucket_far, action, diff,
                     robot, last_bucket_near):
    """
    Shaped reward function.

    Rewards:  forward progress, lane centering, smooth arcs.
    Penalties: sharp pivots when centered, wrong-direction turns, being lost.
    """
    reward = 0.0

    # Detect an upcoming sharp turn from mid/far sensors
    sharp_turn_detected = False
    turn_direction = 0  # -1 = left, +1 = right

    if bucket_mid != 9 and bucket_far != 9:
        if abs(bucket_mid - 4) >= 3 or abs(bucket_far - 4) >= 4:
            sharp_turn_detected = True
            turn_direction = 1 if (bucket_mid > 4 or bucket_far > 4) else -1

    # 1. Progress along the track
    if diff > 0:
        reward += 10.0
        if action in [0, 3, 4]:
            reward += 2.0
    elif diff == 0:
        reward -= 3.0
    else:
        reward -= 15.0

    # 2. Lateral positioning relative to track center (bucket 4)
    if bucket_near == 9:  # Line lost
        reward -= 8.0
        if action in [1, 2]:
            reward += 2.0

    elif bucket_near == 4:  # Perfectly centered
        if sharp_turn_detected:
            if action in [1, 2]:
                correct = (turn_direction == -1 and action == 1) or \
                          (turn_direction ==  1 and action == 2)
                reward += 8.0 if correct else -5.0
            elif action in [3, 4]:
                correct = (turn_direction == -1 and action == 3) or \
                          (turn_direction ==  1 and action == 4)
                reward += 6.0 if correct else -3.0
            else:
                reward += 3.0
        else:
            if   action == 0:        reward += 15.0
            elif action in [3, 4]:   reward +=  2.0
            else:                    reward -=  8.0

    elif bucket_near in [3, 5]:  # Slightly off-center
        if sharp_turn_detected:
            if action in [1, 2]:
                correct = ((bucket_near == 3 or turn_direction == -1) and action == 1) or \
                          ((bucket_near == 5 or turn_direction ==  1) and action == 2)
                reward += 10.0 if correct else -6.0
            elif action in [3, 4]:
                correct = (bucket_near == 3 and action == 4) or \
                          (bucket_near == 5 and action == 3)
                reward += 7.0 if correct else -5.0
        else:
            if action == 0:
                reward += 8.0
            elif action in [3, 4]:
                correct = (bucket_near == 3 and action == 4) or \
                          (bucket_near == 5 and action == 3)
                reward += 12.0 if correct else -4.0
            else:
                reward -= 5.0

    elif bucket_near in [2, 6]:  # Moderately off-center
        if sharp_turn_detected:
            if action in [1, 2]:
                correct = (bucket_near == 2 and action == 2) or \
                          (bucket_near == 6 and action == 1)
                reward += 8.0 if correct else -7.0
            elif action in [3, 4]:
                correct = (bucket_near == 2 and action == 4) or \
                          (bucket_near == 6 and action == 3)
                reward += 5.0 if correct else -6.0
        else:
            if action in [3, 4]:
                correct = (bucket_near == 2 and action == 4) or \
                          (bucket_near == 6 and action == 3)
                reward += 8.0 if correct else -6.0
            elif action in [1, 2]:
                correct = (bucket_near == 2 and action == 2) or \
                          (bucket_near == 6 and action == 1)
                reward += 3.0 if correct else -7.0
            else:
                reward -= 6.0

    elif bucket_near in [1, 7]:  # Severely off-center
        if action in [1, 2]:
            correct = (bucket_near == 1 and action == 2) or \
                      (bucket_near == 7 and action == 1)
            reward += 6.0 if correct else -8.0
        elif action in [3, 4]:
            correct = (bucket_near == 1 and action == 4) or \
                      (bucket_near == 7 and action == 3)
            reward += 4.0 if correct else -8.0
        else:
            reward -= 9.0

    else:  # Extreme (bucket 0 or 8)
        if action in [1, 2]:
            correct = (bucket_near == 0 and action == 2) or \
                      (bucket_near == 8 and action == 1)
            reward += 3.0 if correct else -10.0
        else:
            reward -= 12.0

    # 3. Centering trend bonus
    if last_bucket_near != 9 and bucket_near != 9:
        old_dist = abs(last_bucket_near - 4)
        new_dist = abs(bucket_near - 4)
        reward += 4.0 if new_dist < old_dist else (-4.0 if new_dist > old_dist else 0)

    # 4. Sharp-turn anticipation bonus
    if sharp_turn_detected:
        if action in [1, 2]:
            reward += 4.0
        if action == 0 and bucket_near not in [3, 4, 5]:
            reward -= 3.0
    else:
        if action in [3, 4] and bucket_near in [3, 4, 5]:
            reward += 2.0

    # 5. Forward-motion efficiency bonus (not during sharp turns)
    if action in [0, 3, 4]:
        if not (sharp_turn_detected and bucket_near not in [3, 4, 5]):
            reward += 1.0

    # 6. Sustained progress bonus
    if action in [0, 3, 4] and robot.steps_forward > 5 and not sharp_turn_detected:
        reward += 3.0

    # 7. Consistency penalty (sudden pivot when well-centered)
    if robot.last_action in [0, 3, 4] and action in [1, 2]:
        if bucket_near in [3, 4, 5] and not sharp_turn_detected:
            reward -= 4.0

    return reward


def main():
    global epsilon, q_table
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Autonomous Robot – Q-Learning Simulator")
    clock = pygame.time.Clock()

    MAX_LOST_FRAMES = 10

    for episode in range(EPISODES):
        track = Track()
        robot = Robot(track)

        done = False
        total_reward = 0
        laps_completed = 0

        discrete_sensors, _ = robot.get_sensors_discrete(track)
        current_state = get_state_tuple(discrete_sensors, robot.gyro_z, robot.last_action)

        current_idx, _ = track.get_closest_index(robot.x, robot.y)
        last_idx = current_idx

        should_render = (episode % SHOW_EVERY == 0)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    np.save(Q_FILE, q_table)
                    pygame.quit()
                    return

            # Epsilon-greedy action selection
            if np.random.random() > epsilon:
                action = np.argmax(q_table[current_state])
            else:
                action = np.random.randint(0, len(ACTIONS))

            accumulated_reward = 0

            for _ in range(STEPS_TO_REPEAT):
                robot.move(action)
                discrete_sensors, _ = robot.get_sensors_discrete(track)

                bucket_near = discrete_sensors[0]
                bucket_mid  = discrete_sensors[1]
                bucket_far  = discrete_sensors[2]

                robot.consecutive_lost = (robot.consecutive_lost + 1
                                          if bucket_near == 9 else 0)

                if robot.consecutive_lost > MAX_LOST_FRAMES:
                    accumulated_reward = -150
                    done = True
                    break

                current_idx, _ = track.get_closest_index(robot.x, robot.y)
                diff = current_idx - last_idx
                if diff < -200: diff = 1
                elif diff > 200: diff = -1

                # Lap completion detection
                if (last_idx > len(track.points) - 20
                        and current_idx < 20 and diff == 1):
                    laps_completed += 1
                    accumulated_reward += 100.0
                    print(f"Episode {episode}: Lap {laps_completed} complete!")
                    if laps_completed >= 3:
                        done = True
                        accumulated_reward += 200
                        print(f"Episode {episode}: TRACK MASTERED!")

                step_reward = calculate_reward(
                    bucket_near, bucket_mid, bucket_far,
                    action, diff, robot, robot.last_bucket_near
                )
                accumulated_reward += step_reward
                robot.last_bucket_near = bucket_near
                last_idx = current_idx

                if should_render:
                    screen.fill((20, 20, 20), (0, 0, SIM_WIDTH, WINDOW_HEIGHT))
                    for p in track.points:
                        pygame.draw.circle(screen, GRAY, p, TRACK_WIDTH // 2)
                    pygame.draw.circle(screen, BLUE,
                                       (int(robot.x), int(robot.y)), 15)
                    ex = robot.x + 20 * math.cos(math.radians(robot.angle))
                    ey = robot.y + 20 * math.sin(math.radians(robot.angle))
                    pygame.draw.line(screen, RED,
                                     (robot.x, robot.y), (ex, ey), 3)
                    draw_combined_vision(screen, robot, track, discrete_sensors)
                    draw_info_panel(screen, episode, total_reward, epsilon,
                                    action, laps_completed, robot.last_action)
                    pygame.display.flip()
                    clock.tick(60)

                if (robot.x < 0 or robot.x > SIM_WIDTH
                        or robot.y < 0 or robot.y > WINDOW_HEIGHT):
                    accumulated_reward = -150
                    done = True
                    break

            # Bellman update
            new_state = get_state_tuple(discrete_sensors, robot.gyro_z, action)
            if not done:
                max_future = np.max(q_table[new_state])
                current_q  = q_table[current_state + (action,)]
                q_table[current_state + (action,)] = (
                    (1 - LEARNING_RATE) * current_q
                    + LEARNING_RATE * (accumulated_reward + DISCOUNT * max_future)
                )
            else:
                q_table[current_state + (action,)] = accumulated_reward

            current_state      = new_state
            robot.last_action  = action
            total_reward      += accumulated_reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 100 == 0:
            np.save(Q_FILE, q_table)
            print(f"Ep {episode} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

    np.save(Q_FILE, q_table)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to exit...")
