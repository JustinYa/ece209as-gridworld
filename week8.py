import random
import numpy as np
import math

# ======================================================================
#  Environment Configuration
# ======================================================================

# Robot initial position
robot_pos = [2, 0]   # Starting at (2,0)

# “True” reward locations (environment uses these)
TRUE_RD = [2, 2]     # Reward = +1
TRUE_RS = [2, 4]     # Reward = +10

# Forbidden states (walls)
forbidden_state = [[1, 1], [2, 1], [1, 3], [2, 3]]

# Walls with negative reward
RW_loc = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]

# Reward values
RW_val = -1
RD_val = 1
RS_val = 10

# Transition noise: probability robot does NOT follow commanded action
Pe = 0.3
gamma = 0.2   # Discount factor for value iteration


# ======================================================================
#  Visualization
# ======================================================================

def display_grid():
    """Print the grid, with robot / reward / forbidden states."""
    print("\n" + "=" * 30)
    for y in range(5):
        row = ""
        for x in range(5):
            pos = [x, y]

            if pos == robot_pos:
                cell = " R "
            elif pos == TRUE_RD:
                cell = "RD "
            elif pos == TRUE_RS:
                cell = "RS "
            elif pos in RW_loc:
                cell = "RW "
            elif pos in forbidden_state:
                cell = " X "
            else:
                cell = " . "
            row += cell
        print(row)
    print(f"Robot position: {tuple(robot_pos)}")
    print("=" * 30)


# ======================================================================
#  Environment Dynamics
# ======================================================================

def move_robot(a):
    """Move robot with error probability Pe."""
    if random.random() < Pe:
        # Pick a random action (not a)
        actions = ["up", "down", "left", "right", "stay"]
        actions.remove(a)
        actual = random.choice(actions)
        print(f"Robot disobeyed! Instead of {a}, it went {actual.upper()}")
    else:
        actual = a
        print(f"Robot followed command: {actual.upper()}")

    global robot_pos
    old_pos = robot_pos.copy()

    if actual == "up" and robot_pos[1] > 0:
        robot_pos[1] -= 1
    elif actual == "down" and robot_pos[1] < 4:
        robot_pos[1] += 1
    elif actual == "left" and robot_pos[0] > 0:
        robot_pos[0] -= 1
    elif actual == "right" and robot_pos[0] < 4:
        robot_pos[0] += 1

    # Block forbidden states
    if robot_pos in forbidden_state:
        robot_pos = old_pos.copy()
        print("Blocked by forbidden state!")


def env_observation():
    """Compute noisy harmonic-mean observation o_t, using TRUE map."""

    curr = np.array(robot_pos)
    RD = np.array(TRUE_RD)
    RS = np.array(TRUE_RS)

    dRD = np.linalg.norm(curr - RD)
    dRS = np.linalg.norm(curr - RS)

    if dRD == 0 or dRS == 0:
        h = 0
    else:
        h = 2 / (1/dRD + 1/dRS)

    ceil_h = np.ceil(h)
    floor_h = np.floor(h)
    prob_floor = ceil_h - h

    if np.random.rand() < prob_floor:
        return int(floor_h)
    return int(ceil_h)


def reward_fn(pos):
    """True environment reward (not belief)."""
    if pos in RW_loc:
        return RW_val
    if pos == TRUE_RD:
        return RD_val
    if pos == TRUE_RS:
        return RS_val
    return 0


# ======================================================================
#  State Transition Probability  P(s'|s,a)
# ======================================================================

def P(next_state, s, a):
    """Transition probability with Pe stochasticity."""

    def step(state, act):
        x, y = state
        nx, ny = x, y

        if act == "up" and y > 0:
            ny -= 1
        elif act == "down" and y < 4:
            ny += 1
        elif act == "left" and x > 0:
            nx -= 1
        elif act == "right" and x < 4:
            nx += 1

        candidate = [nx, ny]
        if candidate in forbidden_state:
            return state
        return candidate

    actions = ["up", "down", "left", "right", "stay"]

    prob = 0.0

    # Follow commanded action
    intended = step(s, a)
    if intended == next_state:
        prob += (1 - Pe)

    # Disobey actions
    others = [act for act in actions if act != a]
    for act in others:
        s2 = step(s, act)
        if s2 == next_state:
            prob += (Pe / len(others))

    return prob


# ======================================================================
#  Value Iteration (for computing optimal policy)
# ======================================================================

def value_iteration():
    """Compute optimal value function V* and policy π*."""
    actions = ["up", "down", "left", "right", "stay"]
    states = [[x, y] for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]

    # Initialize V(s) = 0
    V = {tuple(s): 0.0 for s in states}
    theta = 1e-6

    # Value iteration loop
    while True:
        delta = 0
        newV = {}

        for s in states:
            s_tup = tuple(s)
            Qs = []

            for a in actions:
                exp_val = 0
                for s2 in states:
                    exp_val += P(s2, s, a) * (reward_fn(s) + gamma * V[s2])
                Qs.append(exp_val)

            newV[s_tup] = max(Qs)
            delta = max(delta, abs(V[s_tup] - newV[s_tup]))

        V = newV
        if delta < theta:
            break

    # Extract optimal policy
    policy = {}
    Q_star = {}

    for s in states:
        s_tup = tuple(s)
        best_a = None
        best_val = -1e9

        Q_star[s_tup] = {}

        for a in actions:
            q = 0
            for s2 in states:
                q += P(s2, s, a) * (reward_fn(s) + gamma * V[s2])
            Q_star[s_tup][a] = q

            if q > best_val:
                best_val = q
                best_a = a

        policy[s_tup] = best_a

    return V, Q_star, policy


# ======================================================================
#  Bayesian State Estimation (SLAM-like belief update)
# ======================================================================

# Three assumed maps for testing map error
assumed_maps = {
    "true": {"RD": [2, 2], "RS": [2, 4]},
    "near": {"RD": [1, 2], "RS": [2, 4]},
    "far":  {"RD": [0, 0], "RS": [4, 4]},
}


def P_o_given_s(o, s, assumed_map):
    """Likelihood P(o | s, m_assumed)."""

    s = np.array(s)
    RD = np.array(assumed_map["RD"])
    RS = np.array(assumed_map["RS"])

    dRD = np.linalg.norm(s - RD)
    dRS = np.linalg.norm(s - RS)

    if dRD == 0 or dRS == 0:
        h = 0
    else:
        h = 2 / (1/dRD + 1/dRS)

    ceil_h = np.ceil(h)
    floor_h = np.floor(h)

    if o == ceil_h:
        return 1 - (ceil_h - h)
    if o == floor_h:
        return (ceil_h - h)
    return 0.0


def P_r_given_s(r, s, assumed_map):
    """Likelihood P(r | s, m_assumed)."""

    RD = tuple(assumed_map["RD"])
    RS = tuple(assumed_map["RS"])

    if tuple(s) == RD:
        expected = RD_val
    elif tuple(s) == RS:
        expected = RS_val
    elif list(s) in RW_loc:
        expected = RW_val
    else:
        expected = 0

    return 0.9 if r == expected else 0.1


def Bel_t_update(prev_bel, action, observation, reward, assumed_map):
    """Bayes filter update: P(s|o,r,a)."""

    states = [tuple([x, y]) for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]

    # Prediction
    bel_pred = {}
    for s_next in states:
        total = 0
        for s_prev in states:
            total += P(list(s_next), list(s_prev), action) * prev_bel[s_prev]
        bel_pred[s_next] = total

    # Normalize prediction
    Z1 = sum(bel_pred.values())
    if Z1 > 0:
        for s in bel_pred:
            bel_pred[s] /= Z1

    # Correction
    bel_new = {}
    for s in states:
        bel_new[s] = (
            P_o_given_s(observation, s, assumed_map) *
            P_r_given_s(reward, s, assumed_map) *
            bel_pred[s]
        )

    Z2 = sum(bel_new.values())
    if Z2 > 0:
        for s in bel_new:
            bel_new[s] /= Z2

    return bel_new


def belief_entropy(bel):
    H = 0
    for p in bel.values():
        if p > 0:
            H -= p * math.log(p)
    return H


def belief_map_state(bel):
    return max(bel.items(), key=lambda kv: kv[1])[0]


def reset_robot_and_belief():
    global robot_pos
    robot_pos = [2, 0]

    states = [(x, y) for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]

    bel = {s: 1/len(states) for s in states}
    return bel


# ======================================================================
#  Fixed trajectory evaluation (SLAM experiment)
# ======================================================================

def run_trajectory(map_name, actions):
    """Run same trajectory under different assumed maps."""
    global robot_pos

    print("\n" + "#" * 60)
    print(f"Running trajectory with assumed map: {map_name}")
    print("#" * 60)

    random.seed(0)
    np.random.seed(0)
    bel = reset_robot_and_belief()
    m = assumed_maps[map_name]

    total_reward = 0

    for t, a in enumerate(actions, start=1):

        print(f"\nStep {t}: action = {a}")

        move_robot(a)
        r_t = reward_fn(robot_pos)
        total_reward += r_t
        print(f"Immediate reward r_t = {r_t}")

        o_t = env_observation()
        print(f"Observation o_t = {o_t}")

        bel = Bel_t_update(bel, a, o_t, r_t, m)
        s_MAP = belief_map_state(bel)
        err = math.dist(robot_pos, s_MAP)
        H = belief_entropy(bel)

        print(f"True position: {tuple(robot_pos)}")
        print(f"MAP estimate:  {s_MAP}")
        print(f"L2 error:      {err:.2f}")
        print(f"Entropy:       {H:.3f}")

    print(f"\n==> Final total reward for map {map_name}: {total_reward}")


# ======================================================================
#  Main Execution
# ======================================================================

if __name__ == "__main__":

    print("Running Value Iteration...\n")
    V, Q, pi = value_iteration()

    print("Optimal Value Function:")
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                row += "  X   "
            else:
                row += f"{V[s]:5.2f} "
        print(row)

    print("\nOptimal Policy:")
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                row += "X   "
            else:
                a = pi[s]
                row += a[0].upper() + "   "
        print(row)

    # Fixed test trajectory (same for all maps)
    traj = ["up", "up", "right", "right", "down", "down", "left", "left"]

    run_trajectory("true", traj)
    run_trajectory("near", traj)
    run_trajectory("far", traj)
