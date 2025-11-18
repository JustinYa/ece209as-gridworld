import random
import numpy as np
import math

# ----------------- Global settings -----------------
Pe = 0.3   # robot disobey probability
r = 0      # latest reward

# Robot initial position (center-ish of the grid)
robot_pos = [2, 0]  # [x, y]

# Forbidden and reward states
forbidden_state = [[1, 1], [2, 1], [1, 3], [2, 3]]
RW_loc = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]  # walls with -1 reward

RW_val = -1
RD_val = 1
RS_val = 10

# ---- True map used by the ENVIRONMENT (for observations & rewards) ----
TRUE_MAP = {
    "RD": [2, 2],   # real delicious ice cream
    "RS": [2, 4]    # real super ice cream
}
RD_loc = [TRUE_MAP["RD"]]
RS_loc = [TRUE_MAP["RS"]]

# ---- Assumed maps for the Bayes filter ----
assumed_maps = {
    "true": {"RD": [2, 2], "RS": [2, 4]},      # exactly correct
    "near": {"RD": [1, 2], "RS": [2, 4]},      # slightly wrong
    "far":  {"RD": [0, 0], "RS": [4, 4]}       # very wrong
}


# ----------------- Display functions -----------------
def display_grid():
    """Display the 5x5 grid with robot, rewards, and forbidden states"""
    print("\n" + "=" * 30)
    for y in range(5):
        row = ""
        for x in range(5):
            pos = [x, y]

            if pos == robot_pos:
                cell = " R "
            elif pos in RD_loc:
                cell = "RD "
            elif pos in RS_loc:
                cell = "RS "
            elif pos in RW_loc:
                cell = "RW "
            elif pos in forbidden_state:
                cell = " X "
            else:
                cell = " . "
            row += cell
        print(row)
    print(f"Robot position: ({robot_pos[0]}, {robot_pos[1]})")
    print("=" * 30)


def display_belief(bel):
    """Display belief distribution as a 5x5 grid"""
    print("\nBelief Distribution (P(s)):")
    print("=" * 40)
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                cell = "  X   "
            else:
                prob = bel.get(s, 0.0)
                cell = f"{prob:5.2f} "
            row += cell
        print(row)
    print("=" * 40)


# ----------------- Dynamics -----------------
def move_from_state(state, action):
    """Deterministic move from ANY state with boundary + forbidden checks."""
    x, y = state
    new_x, new_y = x, y

    if action == "up" and y > 0:
        new_y -= 1
    elif action == "down" and y < 4:
        new_y += 1
    elif action == "left" and x > 0:
        new_x -= 1
    elif action == "right" and x < 4:
        new_x += 1
    # "stay" keeps position

    new_state = [new_x, new_y]

    # bump off forbidden states: if next is forbidden, stay in original state
    if new_state in forbidden_state:
        return state.copy()
    else:
        return new_state


def move_robot(a):
    """Move the global robot_pos with 30% chance of not following user input."""
    global robot_pos

    if random.random() < Pe:
        possible_actions = ["up", "down", "left", "right", "stay"]
        possible_actions.remove(a)
        actual_direction = random.choice(possible_actions)
        print(f"Robot disobeyed! Instead of {a.upper()}, it went {actual_direction.upper()}")
    else:
        actual_direction = a
        print(f"Robot followed command: {actual_direction.upper()}")

    robot_pos[:] = move_from_state(robot_pos, actual_direction)


def P(next_state, s_prev, a):
    """
    Transition probability P(next_state | s_prev, a)
    Uses same noise model as move_robot:
      - obey command with prob 1 - Pe
      - otherwise choose a different action uniformly from 4 others
    """
    actions = ["up", "down", "left", "right", "stay"]

    # obey
    intended = move_from_state(s_prev, a)
    prob = 0.0
    if next_state == intended:
        prob += (1 - Pe)

    # disobey: pick among other 4 actions
    other_actions = [act for act in actions if act != a]
    for act in other_actions:
        s_alt = move_from_state(s_prev, act)
        if next_state == s_alt:
            prob += Pe / 4.0

    return prob


# ----------------- Observation model -----------------
def compute_o():
    """
    Environment observation o based on TRUE_MAP (real RD/RS).
    """
    curr_pos = np.array(robot_pos)
    R_D_pos = np.array(TRUE_MAP["RD"])
    R_S_pos = np.array(TRUE_MAP["RS"])

    d_D = np.linalg.norm(curr_pos - R_D_pos)
    d_S = np.linalg.norm(curr_pos - R_S_pos)

    if d_D == 0 or d_S == 0:
        h = 0
    else:
        h = 2 / (1 / d_D + 1 / d_S)

    ceil_h = np.ceil(h)
    floor_h = np.floor(h)
    prob_floor = ceil_h - h  # probability of rounding DOWN

    if np.random.rand() < prob_floor:
        o = int(floor_h)
    else:
        o = int(ceil_h)
    return o


def P_o_given_s(o, s, assumed_map):
    """
    P(o | s, assumed_map).
    assumed_map is a dict: {"RD": [x,y], "RS": [x,y]}
    """
    curr_pos = np.array(s)
    R_D_pos = np.array(assumed_map["RD"])
    R_S_pos = np.array(assumed_map["RS"])

    d_D = np.linalg.norm(curr_pos - R_D_pos)
    d_S = np.linalg.norm(curr_pos - R_S_pos)

    if d_D == 0 or d_S == 0:
        h = 0
    else:
        h = 2 / (1 / d_D + 1 / d_S)

    ceil_h = np.ceil(h)
    floor_h = np.floor(h)

    if np.isclose(o, ceil_h):
        return 1 - (ceil_h - h)
    elif np.isclose(o, floor_h):
        return (ceil_h - h)
    else:
        return 0.0


# ----------------- Rewards -----------------
def check_rewards():
    """Check if the robot has reached a reward location and update total reward."""
    global r

    if robot_pos in RW_loc:
        r = RW_val
        print(f"Robot reached RW location! Reward = {RW_val}.")
    elif robot_pos in RD_loc:
        r = RD_val
        print(f"Robot reached RD location! Reward = {RD_val}.")
    elif robot_pos in RS_loc:
        r = RS_val
        print(f"Robot reached RS location! Reward = {RS_val}.")


# ----------------- Belief update -----------------
def Bel_t_update(prev_bel, action, observation, assumed_map):
    """
    One-step Bayes filter update:
        prediction using P(s_t | s_{t-1}, a)
        correction using P(o_t | s_t, assumed_map)
    """
    states = [[x, y] for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]

    # prediction
    bel_pred = {}
    for s_next in states:
        total = 0.0
        for s_prev in states:
            total += P(s_next, s_prev, action) * prev_bel[tuple(s_prev)]
        bel_pred[tuple(s_next)] = total

    # normalize
    norm_pred = sum(bel_pred.values())
    if norm_pred > 0:
        for s in bel_pred:
            bel_pred[s] /= norm_pred

    # correction
    bel_new = {}
    for s in states:
        bel_new[tuple(s)] = P_o_given_s(observation, s, assumed_map) * bel_pred[tuple(s)]

    norm = sum(bel_new.values())
    if norm > 0:
        for s in bel_new:
            bel_new[s] /= norm

    return bel_new


# ----------------- Metrics helpers -----------------
def belief_entropy(bel):
    """Shannon entropy of belief distribution."""
    H = 0.0
    for p in bel.values():
        if p > 0:
            H -= p * math.log(p + 1e-12)
    return H


def belief_map_state(bel):
    """Return MAP state (argmax) of belief."""
    return max(bel.items(), key=lambda kv: kv[1])[0]


def l2_error(pos_true, pos_est):
    return math.sqrt((pos_true[0] - pos_est[0]) ** 2 + (pos_true[1] - pos_est[1]) ** 2)


def reset_robot_and_belief():
    """Reset robot pos and uniform belief."""
    global robot_pos
    robot_pos = [2, 0]
    states = [(x, y) for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]
    Bel = {s: 1.0 / len(states) for s in states}
    return Bel


def run_trajectory(map_name, actions):
    """Run a fixed trajectory under a given assumed map and print metrics."""
    print("\n" + "#" * 60)
    print(f"Running trajectory with assumed map: {map_name}")
    print("#" * 60)

    random.seed(0)
    np.random.seed(0)

    assumed_map = assumed_maps[map_name]
    Bel = reset_robot_and_belief()

    for t, a in enumerate(actions, start=1):
        print(f"\nStep {t}: action = {a}")
        move_robot(a)
        check_rewards()
        o = compute_o()
        print(f"Observation o = {o}")

        Bel = Bel_t_update(Bel, a, o, assumed_map)
        H = belief_entropy(Bel)
        s_map = belief_map_state(Bel)
        err = l2_error(robot_pos, s_map)

        print(f"True position: {tuple(robot_pos)}")
        print(f"MAP estimate: {s_map}")
        print(f"L2 error: {err:.2f}")
        print(f"Belief entropy: {H:.3f}")


# ----------------- Interactive main loop -----------------
def interactive_main():
    print("Robot Grid Movement System (interactive)")
    print("WARNING: Robot has 30% chance to disobey commands!")
    print("Commands: up, down, left, right, stay, quit")

    choice = input("Choose assumed map (true / near / far) [true]: ").strip().lower()
    if choice not in assumed_maps:
        choice = "true"
    assumed_map = assumed_maps[choice]
    print(f"Using assumed map: {choice}  -> RD={assumed_map['RD']}, RS={assumed_map['RS']}")

    states = [(x, y) for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]
    Bel = {s: 1.0 / len(states) for s in states}

    while True:
        display_grid()
        display_belief(Bel)

        command = input("\nEnter command: ").strip().lower()
        if command == "quit":
            print("Goodbye!")
            break
        elif command in ["up", "down", "left", "right", "stay"]:
            move_robot(command)
            check_rewards()

            new_o = compute_o()
            print(f"Observation o = {new_o}")

            Bel = Bel_t_update(Bel, command, new_o, assumed_map)
        else:
            print("Invalid command! Use: up, down, left, right, stay, quit")


if __name__ == "__main__":
    traj = ["up", "up", "right", "right", "down", "down", "left", "left"]
    run_trajectory("true", traj)
    run_trajectory("near", traj)
    run_trajectory("far", traj)
