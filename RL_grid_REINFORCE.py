import torch
import torch.nn as nn
import torch.optim as optim
import random

# ===== 环境参数 =====
n_states = 5
ACTIONS = ["left", "right"]
goal_state = 4
gamma = 0.9
episodes = 200

# ===== 环境函数 =====
def take_action(state, action):
    """执行动作并返回(next_state, reward)"""
    if action == "right":
        next_state = min(state + 1, n_states - 1)
    elif action == "left":
        next_state = max(state - 1, 0)
    else:
        raise ValueError("未知动作标签")
    reward = 1 if next_state == goal_state else 0
    return next_state, reward

# ===== 策略网络 =====
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc = nn.Linear(n_states, 16)
        self.out = nn.Linear(16, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.out(x), dim=-1)  # 输出动作概率

# ===== 状态编码 =====
def one_hot_state(state):
    return torch.eye(n_states)[state].unsqueeze(0)  # shape (1, n_states)

# ===== 初始化 =====
policy_net = PolicyNet(n_states, len(ACTIONS))
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# ===== REINFORCE 训练 =====
for ep in range(episodes):
    state = 0
    log_probs = []
    rewards = []

    # 采样一个 episode
    while state != goal_state:
        probs = policy_net(one_hot_state(state))
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        action = ACTIONS[action_idx]

        # 保存 log-prob 用于梯度更新
        log_probs.append(dist.log_prob(torch.tensor(action_idx)))

        # 与环境交互
        next_state, reward = take_action(state, action)
        rewards.append(reward)

        state = next_state

    # 计算每个时间步的 return（折扣累计奖励）
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # 标准化 returns（提升稳定性）
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 策略梯度更新（REINFORCE）
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G   # 梯度上升 → 负损失下降

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== 检查策略输出 =====
for s in range(n_states):
    probs = policy_net(one_hot_state(s)).detach().numpy()[0]
    print(f"State {s}: " + ", ".join(f"{a}={p:.2f}" for a, p in zip(ACTIONS, probs)))
