基于强化学习的动态参考速度规划算法
一、问题定义
1.1 任务目标
在自动驾驶场景中，根据实时交通环境（周围车辆状态、道路条件等），动态生成安全、舒适、高效的参考速度轨迹。
1.2 马尔可夫决策过程（MDP）建模

    状态空间（State Space）：S = {s_t}，包含自车和周围车辆的状态信息
    动作空间（Action Space）：A = {a_t}，连续动作空间，表示期望速度
    奖励函数（Reward Function）：R(s_t, a_t)，评估速度选择的好坏
    转移函数（Transition Function）：P(s_{t+1}|s_t, a_t)，环境动力学
    折扣因子（Discount Factor）：γ ∈ (0, 1)，权衡当前与未来奖励

二、状态空间定义
2.1 状态向量组成
状态向量 s_t 包含以下信息：
文本

s_t = [ego_state, neighbor_states]

其中：
ego_state = [v_ego, a_ego, pos_x, pos_y, heading]
  - v_ego: 自车速度 (m/s)
  - a_ego: 自车加速度 (m/s²)
  - pos_x, pos_y: 自车位置坐标 (m)
  - heading: 自车航向角 (rad)

neighbor_states = [v_neighbor_1, a_neighbor_1, rel_pos_x_1, rel_pos_y_1, ...,
                   v_neighbor_N, a_neighbor_N, rel_pos_x_N, rel_pos_y_N]
  - v_neighbor_i: 邻居车辆i的速度 (m/s)
  - a_neighbor_i: 邻居车辆i的加速度 (m/s²)
  - rel_pos_x_i, rel_pos_y_i: 相对于自车的相对位置 (m)
  - N: 考虑的邻居车辆数量（通常取4-8辆）

2.2 状态归一化
为提高训练稳定性，对状态进行归一化处理：
文本

s_normalized = (s - s_mean) / s_std

三、动作空间定义
3.1 动作范围
文本

a_t ∈ [0, v_base]

    v_base: 基础速度上限（例如高速公路场景为30 m/s，城市道路为15 m/s）
    动作类型: 连续值，表示期望的纵向速度

3.2 动作约束
为保证物理可行性，添加加速度和减速度约束：
文本

a_min ≤ (a_t - v_ego) / Δt ≤ a_max

    a_min: 最大减速度（通常-5 m/s²）
    a_max: 最大加速度（通常3 m/s²）
    Δt: 控制周期（通常0.1 s）

四、奖励函数设计
4.1 奖励函数公式
文本

R_speed = ω₁·F_TTC + ω₂·F_jerk + ω₃·F_dis

其中 ω₁ = ω₂ = ω₃ = 1/3
4.2 特征函数详细计算
4.2.1 碰撞时间特征（F_TTC）
计算公式：
文本

TTC(t) = ‖p_n-1(t) - p_n(t)‖₂ / [(v_n-1(t) - v_n(t))ᵀ · ê_rel]

参数说明：

    p_n-1(t): 前车位置向量
    p_n(t): 自车位置向量
    v_n-1(t): 前车速度向量
    v_n(t): 自车速度向量
    ê_rel: 相对位置单位向量 = (p_n-1 - p_n) / ‖p_n-1 - p_n‖

特征函数：
文本

F_TTC = {
    log(TTC / 4),  if 0 ≤ TTC ≤ 4
    0,              otherwise
}

安全阈值： 4秒（当TTC < 4秒时产生负奖励）
4.2.2 冲击度特征（F_jerk）
冲击度定义：
文本

j(t) = da(t)/dt = d³s(t)/dt³

离散近似计算：
文本

j(t) ≈ (a_t - a_{t-1}) / Δt

特征函数：
文本

F_jerk(t) = -(j(t) / j_max)²

参数：

    j_max = 120 m/s³（最大允许冲击度）
    冲击度越小，舒适性越好，奖励越高

4.2.3 安全距离特征（F_dis）
安全距离计算：
文本

d_safe = V_n × RT + (V_n² / 2a_max) - (V_{n-1}² / 2a_max)

参数说明：

    V_n: 自车速度
    V_{n-1}: 前车速度
    RT: 后车反应时间 = 0.5 s
    a_max: 最大绝对减速度 = 5 m/s²

实际距离：
文本

d_actual = min_j∈N ‖p_j(t) - p_n(t)‖₂

（N为所有邻居车辆集合）
特征函数：
文本

F_dis = d_actual - d_safe

解释：

    F_dis > 0: 安全距离充足，正奖励
    F_dis < 0: 安全距离不足，负奖励

五、DDPG算法实现
5.1 网络架构
5.1.1 Actor网络（策略网络）
文本

输入: 状态向量 s (维度: state_dim)
输出: 动作 a (维度: 1)

网络结构:
Input Layer (state_dim)
    ↓
Dense Layer 1: 400 units + ReLU activation
    ↓
Dense Layer 2: 300 units + ReLU activation
    ↓
Output Layer: 1 unit + tanh activation
    ↓
Scale to [0, v_base]

输出处理：
文本

a_raw = tanh(network_output)  # 范围 [-1, 1]
a_scaled = (a_raw + 1) / 2 * v_base  # 范围 [0, v_base]

5.1.2 Critic网络（价值网络）
文本

输入: 状态向量 s (维度: state_dim) + 动作 a (维度: 1)
输出: Q值 (标量)

网络结构:
Input Layer (state_dim + 1)
    ↓
Dense Layer 1: 400 units + ReLU activation
    ↓
Dense Layer 2: 300 units + ReLU activation
    ↓
Output Layer: 1 unit (linear activation)

输入拼接：
文本

input = concat(s, a)

5.1.3 目标网络
为每个网络创建目标网络：

    Target_Actor (θ_u')
    Target_Critic (θ_Q')

初始化：
文本

θ_u' ← θ_u
θ_Q' ← θ_Q

5.2 算法伪代码
python

# ==================== 初始化 ====================

# 1. 网络初始化
Actor = NeuralNetwork(input_dim=state_dim, output_dim=1, hidden_layers=[400, 300])
Critic = NeuralNetwork(input_dim=state_dim+1, output_dim=1, hidden_layers=[400, 300])

# 2. 目标网络初始化
Target_Actor = copy.deepcopy(Actor)
Target_Critic = copy.deepcopy(Critic)

# 3. 经验回放池
Replay_Buffer = []

# 4. 超参数设置
γ = 0.99          # 折扣因子
τ = 0.001         # 软更新系数
α_actor = 1e-4    # Actor学习率
α_critic = 1e-3   # Critic学习率
batch_size = 64   # 批量大小
max_episodes = 10000  # 最大训练回合数
max_steps = 1000  # 每回合最大步数
noise_std = 0.1   # 探索噪声标准差

# 5. 优化器
Actor_Optimizer = Adam(Actor.parameters, lr=α_actor)
Critic_Optimizer = Adam(Critic.parameters, lr=α_critic)

# ==================== 训练循环 ====================

for episode in range(max_episodes):
    
    # 重置环境
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        
        # ========== 1. 动作选择 ==========
        
        # 确定性策略输出
        action_deterministic = Actor.forward(state)
        
        # 添加探索噪声（Ornstein-Uhlenbeck过程或高斯噪声）
        noise = np.random.normal(0, noise_std)
        action = action_deterministic + noise
        
        # 约束动作范围
        action = np.clip(action, 0, v_base)
        
        # ========== 2. 环境交互 ==========
        
        # 执行动作，获取下一状态和奖励
        next_state, reward, done, info = env.step(action)
        
        # 计算奖励（详细计算见第四章）
        reward = calculate_reward(state, action, next_state)
        
        # 存储经验
        Replay_Buffer.append((state, action, reward, next_state, done))
        
        # 更新累计奖励
        episode_reward += reward
        
        # ========== 3. 网络更新（每步或定期） ==========
        
        if len(Replay_Buffer) >= batch_size:
            
            # 从经验回放池随机采样
            batch = random.sample(Replay_Buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # 转换为张量
            states = np.array(states)
            actions = np.array(actions).reshape(-1, 1)
            rewards = np.array(rewards).reshape(-1, 1)
            next_states = np.array(next_states)
            dones = np.array(dones).reshape(-1, 1)
            
            # ========== 3.1 Critic网络更新 ==========
            
            # 目标Actor生成下一个动作
            next_actions = Target_Actor.forward(next_states)
            
            # 目标Critic评估下一个状态-动作对
            target_Q_values = Target_Critic.forward(next_states, next_actions)
            
            # 计算目标Q值（考虑终止状态）
            target_Q = rewards + γ * target_Q_values * (1 - dones)
            
            # Critic预测当前Q值
            current_Q = Critic.forward(states, actions)
            
            # Critic损失（均方误差）
            critic_loss = MSE(current_Q, target_Q)
            
            # 反向传播更新Critic
            Critic_Optimizer.zero_grad()
            critic_loss.backward()
            Critic_Optimizer.step()
            
            # ========== 3.2 Actor网络更新 ==========
            
            # Actor生成动作
            predicted_actions = Actor.forward(states)
            
            # Critic评估这些动作的Q值
            Q_values = Critic.forward(states, predicted_actions)
            
            # Actor损失（最大化Q值，等价于最小化负Q值）
            actor_loss = -mean(Q_values)
            
            # 反向传播更新Actor
            Actor_Optimizer.zero_grad()
            actor_loss.backward()
            Actor_Optimizer.step()
            
            # ========== 3.3 软更新目标网络 ==========
            
            # 更新目标Critic
            for param, target_param in zip(Critic.parameters, Target_Critic.parameters):
                target_param.data = τ * param.data + (1 - τ) * target_param.data
            
            # 更新目标Actor
            for param, target_param in zip(Actor.parameters, Target_Actor.parameters):
                target_param.data = τ * param.data + (1 - τ) * target_param.data
        
        # 更新状态
        state = next_state
        
        # 检查回合结束
        if done:
            break
    
    # 打印训练信息
    print(f"Episode {episode}, Total Reward: {episode_reward}")

# ==================== 推理/测试 ====================

def inference(state):
    """
    使用训练好的策略进行推理
    """
    # 确定性策略输出（不加噪声）
    action = Actor.forward(state)
    
    # 约束动作范围
    action = np.clip(action, 0, v_base)
    
    return action

六、奖励函数详细实现
python

def calculate_reward(state, action, next_state):
    """
    计算速度规划的奖励函数
    R_speed = ω₁·F_TTC + ω₂·F_jerk + ω₃·F_dis
    """
    
    # 权重
    ω1 = ω2 = ω3 = 1/3
    
    # ========== 1. 计算F_TTC ==========
    
    # 提取自车和邻居车辆信息
    ego_pos = state[2:4]  # [pos_x, pos_y]
    ego_vel = state[0]    # v_ego
    
    F_TTC = 0
    min_TTC = float('inf')
    
    # 遍历所有邻居车辆
    for i in range(num_neighbors):
        neighbor_start = ego_state_dim + i * neighbor_state_dim
        
        neighbor_pos = state[neighbor_start+2:neighbor_start+4]
        neighbor_vel = state[neighbor_start]
        
        # 计算相对位置和速度
        rel_pos = neighbor_pos - ego_pos
        rel_vel = neighbor_vel - ego_vel
        
        # 计算TTC
        distance = np.linalg.norm(rel_pos)
        
        if distance > 0:
            rel_pos_unit = rel_pos / distance
            closing_speed = np.dot(rel_vel, rel_pos_unit)
            
            if closing_speed > 0:  # 车辆正在接近
                TTC = distance / closing_speed
                
                if TTC < min_TTC:
                    min_TTC = TTC
    
    # TTC特征函数
    if 0 <= min_TTC <= 4:
        F_TTC = np.log(min_TTC / 4)
    else:
        F_TTC = 0
    
    # ========== 2. 计算F_jerk ==========
    
    # 从状态中提取当前和前一时刻的加速度
    # 假设状态中包含历史信息，或者从环境获取
    current_accel = calculate_acceleration(state, action)
    
    # 获取前一时刻加速度（需要存储历史）
    prev_accel = get_previous_acceleration()
    
    # 计算冲击度
    jerk = (current_accel - prev_accel) / Δt
    
    # 冲击度特征
    j_max = 120  # m/s³
    F_jerk = -(jerk / j_max) ** 2
    
    # ========== 3. 计算F_dis ==========
    
    # 计算实际最小距离
    min_distance = float('inf')
    
    for i in range(num_neighbors):
        neighbor_start = ego_state_dim + i * neighbor_state_dim
        neighbor_pos = state[neighbor_start+2:neighbor_start+4]
        
        distance = np.linalg.norm(neighbor_pos - ego_pos)
        
        if distance < min_distance:
            min_distance = distance
    
    # 计算安全距离
    RT = 0.5  # 反应时间 (s)
    a_max = 5  # 最大减速度 (m/s²)
    
    d_safe = ego_vel * RT + (ego_vel**2 / (2 * a_max))
    
    # 安全距离特征
    F_dis = min_distance - d_safe
    
    # ========== 4. 组合奖励 ==========
    
    R_speed = ω1 * F_TTC + ω2 * F_jerk + ω3 * F_dis
    
    return R_speed


def calculate_acceleration(state, action):
    """
    根据当前速度和期望速度计算加速度
    """
    current_speed = state[0]
    desired_speed = action
    
    accel = (desired_speed - current_speed) / Δt
    return accel

七、网络实现细节
7.1 Actor网络实现
python

class ActorNetwork:
    def __init__(self, state_dim, action_dim, hidden_layers=[400, 300]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        
        # 初始化权重（Xavier初始化）
        self.weights = []
        self.biases = []
        
        # 输入层到第一隐藏层
        input_size = state_dim
        for hidden_size in hidden_layers:
            W = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
            b = np.zeros(hidden_size)
            self.weights.append(W)
            self.biases.append(b)
            input_size = hidden_size
        
        # 最后一层到输出层
        W_out = np.random.randn(hidden_layers[-1], action_dim) * np.sqrt(2.0 / (hidden_layers[-1] + action_dim))
        b_out = np.zeros(action_dim)
        self.weights.append(W_out)
        self.biases.append(b_out)
    
    def forward(self, state):
        """
        前向传播
        """
        x = state
        
        # 隐藏层
        for i in range(len(self.hidden_layers)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)  # ReLU activation
        
        # 输出层
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        x = np.tanh(x)  # tanh activation
        
        return x
    
    def get_parameters(self):
        return self.weights + self.biases

7.2 Critic网络实现
python

class CriticNetwork:
    def __init__(self, state_dim, action_dim, hidden_layers=[400, 300]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        
        # 输入维度 = 状态维度 + 动作维度
        input_size = state_dim + action_dim
        
        # 初始化权重
        self.weights = []
        self.biases = []
        
        for hidden_size in hidden_layers:
            W = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
            b = np.zeros(hidden_size)
            self.weights.append(W)
            self.biases.append(b)
            input_size = hidden_size
        
        # 输出层
        W_out = np.random.randn(hidden_layers[-1], 1) * np.sqrt(2.0 / (hidden_layers[-1] + 1))
        b_out = np.zeros(1)
        self.weights.append(W_out)
        self.biases.append(b_out)
    
    def forward(self, state, action):
        """
        前向传播
        """
        # 拼接状态和动作
        x = np.concatenate([state, action], axis=-1)
        
        # 隐藏层
        for i in range(len(self.hidden_layers)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)  # ReLU activation
        
        # 输出层（线性激活）
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        
        return x
    
    def get_parameters(self):
        return self.weights + self.biases

八、完整训练流程总结
8.1 训练阶段

    初始化：创建Actor、Critic及其目标网络
    探索：通过添加噪声探索环境
    采样：与环境交互，收集经验到回放池
    学习：
        从回放池随机采样批量数据
        更新Critic网络（最小化TD误差）
        更新Actor网络（最大化Q值）
        软更新目标网络
    迭代：重复步骤2-4直到收敛

8.2 推理阶段

    加载模型：加载训练好的Actor网络
    观测状态：获取当前环境状态
    生成动作：Actor网络输出确定性动作
    执行动作：控制车辆按期望速度行驶
    循环：持续观测-决策-执行

九、关键参数建议
表格
参数  建议值 说明
γ (折扣因子)    0.99    重视长期奖励
τ (软更新系数)   0.001   目标网络缓慢更新
α_actor (Actor学习率)  1e-4    较小学习率保证稳定
α_critic (Critic学习率)    1e-3    较大学习率快速学习
batch_size  64  批量大小
replay_buffer_size  1e6 经验回放池大小
noise_std   0.1 探索噪声标准差
v_base  15-30   基础速度上限（根据场景）
十、注意事项

    状态归一化：训练前对状态进行归一化处理
    动作约束：确保输出动作在物理可行范围内
    目标网络：必须使用目标网络计算TD目标
    经验回放：打破样本相关性，提高训练稳定性
    探索策略：训练初期需要足够探索，后期逐渐减少噪声
    奖励缩放：确保各项特征奖励量级相近