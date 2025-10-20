import random

# 环境参数
n_states = 5
ACTIONS = ["left", "right"]  # 明确动作标签
goal_state = 4

# Q表初始化：状态 × 动作
Q = {state: {action: 0.0 for action in ACTIONS} for state in range(n_states)}

# Q-learning 参数
alpha = 0.1    # 学习率
gamma = 0.9    # 折扣率
epsilon = 0.1  # 探索概率

# 环境函数
def take_action(state, action):
    """执行动作，返回(next_state, reward)"""
    if action == "right":
        next_state = min(state + 1, n_states - 1)
    elif action == "left":
        next_state = max(state - 1, 0)
    else:
        raise ValueError("未知动作")
    reward = 1 if next_state == goal_state else 0
    return next_state, reward

# 训练
for episode in range(100):
    state = 0
    while state != goal_state:
        # ε-greedy 选择动作
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            # 选 Q值最大的动作
            action = max(Q[state], key=Q[state].get)

        # 在环境中执行动作
        next_state, reward = take_action(state, action)

        # Q 更新公式
        best_next_action = max(Q[next_state], key=Q[next_state].get)
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state

# 查看学习结果
for s in range(n_states):
    print(f"State {s}: {Q[s]}")

