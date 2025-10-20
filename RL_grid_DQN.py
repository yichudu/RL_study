import torch
import torch.nn as nn
import torch.optim as optim
import random

# ===== 环境参数 =====
n_states = 5
ACTIONS = ["left", "right"]  # 动作标签
goal_state = 4  # 最后一个格子, 为出口

gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
lr = 0.01  # DQN学习率
episodes = 100


# ===== 环境函数 =====
def take_action(state, action):
    """执行动作并返回 (next_state, reward)"""
    if action == "right":
        next_state = min(state + 1, n_states - 1)
    elif action == "left":
        next_state = max(state - 1, 0)
    else:
        raise ValueError("未知动作")
    reward = 1 if next_state == goal_state else 0
    return next_state, reward


# ===== DQN 网络 =====
class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()
        self.fc = nn.Linear(n_states, 16)  # 输入 -> 隐藏层
        self.out = nn.Linear(16, n_actions)  # 隐藏层 -> 动作Q值

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)  # 输出（每个动作对应的Q值）


# ===== 工具函数 =====
def one_hot_state(state):
    """将状态编码成 One-hot"""
    res = torch.eye(n_states)[state] # Identity Matrix
    res = res.unsqueeze(0)  # shape (1, n_states)
    return res


# ===== 初始化网络和优化器 =====
q_net = QNet(n_states, len(ACTIONS))
optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ===== 训练 =====
for ep in range(episodes):
    state = 0
    while state != goal_state:
        # ε-greedy 选动作
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            with torch.no_grad():
                q_values = q_net(one_hot_state(state))  # 第一次调用, 选最优 action
                action_idx = torch.argmax(q_values).item()
                action = ACTIONS[action_idx]

        # 所选动作对应的Q值
        q_values = q_net(one_hot_state(state)) # 第二次调用, 获取动作对应的Q值
        q_value = q_values[0, ACTIONS.index(action)] # 0 对应 batch_size=1

        # 与环境交互
        next_state, reward = take_action(state, action)

        # TD目标
        with torch.no_grad():
            next_q_values = q_net(one_hot_state(next_state)) # 第三次调用, 获取 next_state 的最大Q值
            best_next_q = torch.max(next_q_values)
            td_target = reward + gamma * best_next_q

        # 计算损失并更新
        loss = loss_fn(q_value, td_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

# ===== 查看训练结果 =====
for s in range(n_states):
    q_val = q_net(one_hot_state(s)).detach().numpy()
    print(f"State {s}: " + ", ".join(f"{action}={q:.2f}" for action, q in zip(ACTIONS, q_val[0])))

"""
State 0: left=1.57, right=1.66
State 1: left=1.79, right=1.81
State 2: left=1.87, right=2.01
State 3: left=1.84, right=2.25
State 4: left=1.28, right=1.38
"""
