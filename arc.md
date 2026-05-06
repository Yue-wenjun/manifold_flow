# Manifold Flow v2.0 项目架构文档

> **核心设计哲学**：计算与渲染解耦。底层强制使用 $N$ 维张量进行真实的数学积分计算（支持 ODE 与 SDE），前端仅接收经过数学降维后的 3 维流形投影数据。

---

## 目录结构总览

```text
manifold_flow/
├── core/
│   ├── types.py          # 全局类型别名（StateVector, Projection3D, Time 等）
│   └── base_system.py    # 抽象基类（DeterministicSystem, StochasticSystem, Projectable3D）
├── solvers/
│   ├── base_solver.py    # ODESolver / SDESolver 抽象接口 + SolverResult
│   ├── rk4_solver.py     # 4阶 Runge-Kutta ODE 积分器
│   └── euler_maruyama.py # Euler-Maruyama SDE 积分器
├── systems/
│   ├── classical.py      # 经典混沌吸引子（Lorenz / Rössler / Chua）
│   ├── shape.py          # 几何形状吸引子（Torus / Ring / Point / Line / Discrete）
│   ├── diffusion.py      # 扩散模型系统（ForwardSDE / ReverseSDE / ProbabilityFlowODE）
│   ├── manifold.py       # 流形学习动力学（t-SNE / UMAP）
│   ├── neural.py         # 神经网络动力学（CANDY / NeuralODE / Hopfield / Transformer / UNet / CANDYDiffusion）
│   └── registry.py       # 全局系统注册表（SystemRegistry + 便捷函数）
├── static/
│   └── js/
│       └── visualization_hybrid.js   # 前端 WebGL/Canvas 渲染逻辑
├── templates/
│   ├── index.html                    # 主控制面板页面
│   └── visualization_hybrid.html    # v2.0 混合可视化页面
└── web/
    ├── app.py            # Flask 应用工厂 + REST API 路由
    └── websocket.py      # WebSocket 流式管理器（WebSocketManager）
run_server.py             # 服务器启动入口
experiments/
├── candy_diffusion.py    # CANDY 扩散实验脚本
├── run_diffusion.py      # 扩散系统批量运行脚本
└── test.py               # 实验性测试
```

---

## 核心抽象层（`core/`）

### `types.py` — 类型基建

| 类型别名 | 底层类型 | 语义 |
|---|---|---|
| `StateVector` | `np.ndarray` | $N$ 维内部状态向量，用于所有数学计算 |
| `Projection3D` | `np.ndarray` | 形状 `(3,)` 或 `(N, 3)` 的前端投影坐标 |
| `ParameterValue` | `float \| int \| np.ndarray` | 系统参数的值类型 |
| `ParameterSet` | `Dict[str, ParameterValue]` | 系统参数字典 |
| `Time` | `float` | 时间标量 |

### `base_system.py` — 抽象基类层级

```
Projectable3D (ABC)
    └── project_to_3d(state) -> Projection3D   ← 强制实现的降维接口

DynamicalSystem(Projectable3D)
    ├── state_dim: int
    ├── parameters: ParameterSet
    ├── get_initial_conditions() -> StateVector  ← 抽象
    └── update_parameters(new_params)

DeterministicSystem(DynamicalSystem)            ← ODE 系统
    └── drift(t, y) -> StateVector              ← 抽象，计算 dy/dt

StochasticSystem(DynamicalSystem)               ← SDE 系统
    ├── drift(t, y) -> StateVector              ← 抽象，确定性漂移项
    └── diffusion(t, y) -> StateVector          ← 抽象，随机扩散项
```

**设计约束**：任何子类若未实现 `project_to_3d`，Python 将在实例化时抛出 `TypeError`，从根本上杜绝了高维数据直接流入前端的可能。

---

## 数值求解器层（`solvers/`）

### `base_solver.py` — 求解器接口

```python
class SolverResult:
    times: np.ndarray          # 形状 (num_steps+1,)
    states: np.ndarray         # 形状 (num_steps+1, state_dim)

class ODESolver(ABC):
    solve(system, y0, t_span, dt) -> SolverResult

class SDESolver(ABC):
    solve(system, y0, t_span, dt) -> SolverResult
```

### `rk4_solver.py` — 4阶 Runge-Kutta

适用于 `DeterministicSystem`，精度 $\mathcal{O}(h^4)$。

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

其中 $k_1 = f(t_n, y_n)$，$k_2 = f(t_n + h/2,\ y_n + hk_1/2)$，以此类推。

### `euler_maruyama.py` — Euler-Maruyama

适用于 `StochasticSystem`，实现 Itô SDE 的一阶强收敛积分：

$$y_{n+1} = y_n + f(t_n, y_n)\,\Delta t + g(t_n, y_n)\,\Delta W_n$$

其中 $\Delta W_n \sim \mathcal{N}(0, \Delta t)$，通过 `np.random.normal(0, 1) * sqrt(dt)` 生成。

> **WebSocket 单步版本**：`websocket.py` 中内联了 `_rk4_step` 和 `_euler_maruyama_step`，用于流式场景下的逐帧积分，避免批量求解的内存开销。

---

## 动力学系统层（`systems/`）

### `classical.py` — 经典混沌吸引子

所有系统均为 **N-Body 粒子系综**（默认 1000 粒子），`state_dim = num_particles × 3`，通过向量化 NumPy 运算并发计算所有粒子的轨迹，展示蝴蝶效应。

| 系统 | 方程 | 默认参数 |
|---|---|---|
| `LorenzSystem` | $\dot{x}=\sigma(y-x),\ \dot{y}=x(\rho-z)-y,\ \dot{z}=xy-\beta z$ | $\sigma=10,\ \rho=28,\ \beta=8/3$ |
| `RosslerSystem` | $\dot{x}=-y-z,\ \dot{y}=x+ay,\ \dot{z}=b+z(x-c)$ | $a=0.2,\ b=0.2,\ c=5.7$ |
| `ChuaSystem` | 含分段线性非线性 $f(x)$ 的三维电路方程 | $\alpha=15.6,\ \beta=28$ |

`project_to_3d` 返回 `(num_particles, 3)` 矩阵，前端将其渲染为粒子点云。

### `shape.py` — 几何形状吸引子

| 系统 | 吸引子类型 | 核心动力学 |
|---|---|---|
| `TorusAttractor` | 环面 | 参数化角速度驱动 + 径向阻尼 |
| `RingAttractor` | 圆环 | 径向收敛力 + 切向旋转流 |
| `PointAttractor` | 点（黑洞） | $\dot{y} = -\alpha y$，指数衰减 |
| `LineAttractor` | 直线 | XY 平面收敛 + Z 轴匀速流动 |
| `DiscreteAttractor` | 跳跃点 | 时变目标点 $\propto \sin(t), \cos(t)$ |

### `diffusion.py` — 扩散模型系统

基于 **VP-SDE（Variance Preserving SDE）** 框架，使用线性噪声调度器：

$$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$$

$$\bar{\alpha}(t) = \exp\!\left(-\int_0^t \beta(s)\,ds\right) = \exp\!\left(-\beta_{\min}t - \frac{\beta_{\max}-\beta_{\min}}{2}t^2\right)$$

| 系统 | 类型 | 方程 |
|---|---|---|
| `ForwardDiffusionSDE` | `StochasticSystem` | $dy = -\frac{1}{2}\beta(t)y\,dt + \sqrt{\beta(t)}\,dW$ |
| `ReverseDiffusionSDE` | `StochasticSystem` | $dy = \left[\frac{1}{2}\beta(\tau)y + \beta(\tau)s_\theta(y,\tau)\right]dt + \sqrt{\beta(\tau)}\,dW$ |
| `ProbabilityFlowODE` | `DeterministicSystem` | $dy = \left[\frac{1}{2}\beta(\tau)y + \frac{1}{2}\beta(\tau)s_\theta(y,\tau)\right]dt$ |

其中 $\tau = 1 - t$ 为反向时间，$s_\theta$ 由 `GMMScoreFunction`（高斯混合模型得分函数）近似，目标分布为 4 个等间距的高斯中心。

**梯度裁剪**：得分函数输出的 $\ell_2$ 范数被限制在 15 以内，防止单步积分将粒子弹飞。

### `manifold.py` — 流形学习动力学

将降维算法建模为连续时间 ODE，`state_dim = N_samples × 3`（直接在 3D 嵌入空间中优化）。

**t-SNE 动力学**（`TSNEDynamicsSystem`）：

$$\frac{dy_i}{dt} = -\eta \cdot \frac{\partial \text{KL}(P \| Q)}{\partial y_i} = -4\eta \sum_j (P_{ij} - Q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$$

- 高维亲和矩阵 $P$ 在初始化时一次性预计算（固定高斯核）
- 低维亲和矩阵 $Q$ 使用 Student-t 分布，每步动态计算
- 前 $t < 2.5$ 时段应用 Early Exaggeration（$\times 4$）

**UMAP 动力学**（`UMAPDynamicsSystem`）：

$$\frac{dy_i}{dt} = -\eta \sum_j \left[ A_{ij} \cdot F_{\text{attr}}(d_{ij}) + (1-A_{ij}) \cdot F_{\text{rep}}(d_{ij}) \right] (y_i - y_j)$$

- 拓扑图 $A$ 通过模糊单纯集（Fuzzy Simplicial Set）近似计算
- 吸引力 $F_{\text{attr}}$ 和排斥力 $F_{\text{rep}}$ 使用 UMAP 标准曲线参数 $(a=1.5769,\ b=0.8950)$

### `neural.py` — 神经网络动力学

所有系统均为 **N-Body 粒子系综**，并发模拟 1000 个网络实例的相空间演化。

| 系统 | 核心机制 | `state_dim` |
|---|---|---|
| `CANDYNetwork` | 分块矩阵乘法 + 自定义激活 $\frac{\|x+1\|-\|x-1\|}{2}$ | `num_particles × hidden_size × 2` |
| `StandardNeuralODE` | 正交权重矩阵 + `tanh` 激活 | `num_particles × state_dim` |
| `HopfieldNetwork` | Hebbian 权重 + `tanh` 动力学，3 个存储模式 | `num_particles × num_neurons` |
| `TransformerAttentionSystem` | 连续时间自注意力 $\dot{Y} = -\gamma Y + \eta \cdot \text{Softmax}(QK^T/\sqrt{d})V$ | `num_particles × embed_dim` |
| `UNetDynamicsSystem` | 编码器-解码器耦合 ODE，跳跃连接作为驱动力 | `num_particles × state_dim × 2` |
| `CANDYDiffusionSystem` | CANDY 掩码 + 图调度因子 $g(t)$ + U-Net 重建流 | `num_particles × state_dim` |

**`CANDYDiffusionSystem` 向量场**（完整公式）：

$$\dot{Y} = -\gamma Y + w_u \cdot \tanh\!\left([(1-g_t)F_{\text{fuse}} + g_t T] W_u^T\right) + w_o \cdot g_t (T - Y)$$

其中 $g_t = \max(0.2,\ 0.7 - \frac{0.5}{T_{\text{param}}} \cdot t)$ 为图调度因子，$T$ 为目标类别中心，$F_{\text{fuse}}$ 为 CANDY 特征与目标的融合输出。

---

## 系统注册表（`systems/registry.py`）

`SystemRegistry` 统一管理所有动力学系统，提供以下接口：

```python
# 便捷函数（模块级）
get_system(name: str, **kwargs) -> DynamicalSystem   # 实例化并覆盖参数
list_systems(category: str = None) -> List[SystemInfo]
get_system_info(name: str) -> SystemInfo
```

已注册系统（按类别）：

| 类别 | 系统 ID |
|---|---|
| `classical` | `lorenz`, `rossler`, `chua` |
| `shape` | `torus`, `ring`, `point`, `line`, `discrete` |
| `diffusion` | `forward_diffusion`, `reverse_diffusion`, `probability_flow` |
| `manifold` | `tsne`, `umap` |
| `neural` | `candy`, `neural_ode`, `hopfield`, `transformer`, `unet`, `candy_diffusion` |

---

## Web 服务层（`web/`）

### `app.py` — Flask 应用工厂

使用工厂模式 `create_app()` 初始化，挂载 CORS 和 WebSocket 管理器。

**REST API 端点**：

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/` | 主控制面板页面 |
| `GET` | `/visualization/hybrid` | v2.0 混合可视化页面 |
| `GET` | `/api/health` | 健康检查，返回版本号 |
| `GET` | `/api/systems` | 获取所有系统列表（支持 `?category=` 过滤） |
| `GET` | `/api/systems/<id>` | 获取特定系统的参数详情（用于生成前端调参滑块） |

### `websocket.py` — 实时流式管理器

**数据流架构**（每帧）：

```
[后端线程]
  N维状态 y (高维张量)
    ↓ solver.step() × steps_per_emit
  更新后的 N维状态 y'
    ↓ system.project_to_3d(y')
  3D投影 proj_3d  (N, 3) 或 (3,)
    ↓ socketio.emit('trajectory_update', payload)
[前端 WebGL]
  接收轻量 JSON，直接渲染
```

**WebSocket 事件协议**：

| 方向 | 事件名 | 数据字段 |
|---|---|---|
| 客户端 → 服务端 | `start_stream` | `stream_id`, `system_id`, `parameters`, `time_step`, `steps_per_emit`, `update_interval` |
| 客户端 → 服务端 | `stop_stream` | `stream_id` |
| 客户端 → 服务端 | `update_parameters` | `stream_id`, `parameters` |
| 服务端 → 客户端 | `trajectory_update` | `stream_id`, `time`, `point` (列表), `is_scatter` (bool) |
| 服务端 → 客户端 | `stream_started` | `stream_id`, `system` |
| 服务端 → 客户端 | `stream_error` | `message` |
| 服务端 → 客户端 | `status` | `message` |

**帧率控制**：后台线程每次计算 `steps_per_emit`（默认 5）步后推送一帧，`emit_interval`（默认 0.033s ≈ 30FPS）控制最高帧率，计算耗时超出时自动跳过 sleep。

**多流并发**：`active_streams` 字典以 `stream_id` 为键管理所有活跃流，每个流运行在独立的 daemon 线程中，`active` 标志位用于优雅停止。

---

## 数据流全链路示意

```
前端用户操作
    │ WebSocket: start_stream { system_id: "lorenz", parameters: {...} }
    ▼
WebSocketManager.start_trajectory_stream()
    │ registry.get_system("lorenz")
    ▼
LorenzSystem (state_dim = 3000)
    │ get_initial_conditions() → y₀ ∈ ℝ³⁰⁰⁰
    ▼
_trajectory_streaming_worker (daemon thread)
    │ 循环: _rk4_step(system, t, y, dt) × 5
    │       → y ∈ ℝ³⁰⁰⁰  (高维积分，绝不出线程)
    │ system.project_to_3d(y)
    │       → proj ∈ ℝ¹⁰⁰⁰ˣ³  (降维，仅此处发生)
    ▼
socketio.emit('trajectory_update', { point: [[x,y,z], ...] })
    │ JSON payload ≈ 24KB (1000点 × 3坐标 × 8字节)
    ▼
前端 WebGL 渲染 (60FPS 目标)
```

---

## 关键设计决策

**1. 计算与渲染严格解耦**

`project_to_3d` 是唯一的降维出口，且只在 `websocket.py` 的 Worker 线程中被调用。高维状态向量 `y` 永远不会序列化传输。

**2. 向量化粒子系综**

所有系统默认模拟 1000 个粒子，利用 NumPy 广播机制在单次矩阵运算中并发计算所有粒子的导数，避免 Python 层面的 for 循环。

**3. ODE/SDE 类型安全**

`ODESolver` 只接受 `DeterministicSystem`，`SDESolver` 只接受 `StochasticSystem`，类型约束在接口层强制执行。WebSocket Worker 通过 `isinstance(system, StochasticSystem)` 动态选择积分器。

**4. 参数热更新**

`update_parameters` 方法允许前端在流运行时实时修改系统参数（如 Lorenz 的 $\rho$），无需重启流，下一积分步即生效。

**5. 数值稳定性保障**

- 扩散系统：得分函数梯度裁剪（$\|\nabla\| \leq 15$），时间下界 $t \geq 0.001$ 防止方差为零
- UMAP：距离平方加 $\epsilon = 10^{-4}$ 防止除零
- t-SNE：Q 矩阵加 $\epsilon = 10^{-12}$ 防止 log(0)
- 反向扩散：$\tau = \max(1-t,\ 0.001)$ 保证 $\beta(\tau) > 0$
